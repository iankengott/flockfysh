from cProfile import label
import os
import shutil
from cv2 import add, reduce
from matplotlib.pyplot import draw
import sys
import torch
import glob
from PIL import Image
import random
import cv2
import gc
import platform


from utilities.output_generation.output_calculations import convert, plot_one_box
from utilities.yolo_utils.yolo_utils import get_exp_dir
from utilities.yolo_utils.train_utils import setup_and_train_yolo, train_yolo
from utilities.optimizers.ram_reducer import reduce_ram_usage
from scraper.DataLoaders.WebDataLoader import WebDataLoader
from config import YOLO_DIR, PHOTO_DIRECTORY, PHOTO_DIRNAME

def run_training_object_detection_webscrape_loop(**args):
	
	#class_names,input_dir,TOTAL_MAXIMUM_IMAGES = 7000, IMAGES_PER_LABEL = 500, MAX_TRAIN_IMAGES = 5000, CONFIDENCE_THRESHOLD = 0.3, SAVE_BB_IMAGE = True, DIM = 200, BATCH = 8, EPOCHS = 50, MAX_TRAINS = 3):

	#TODO: figure out a way to handle params without so much overflow
	MAX_TRAIN_IMAGES = args['images-per-label'] * len(args['class-names'])


	if platform.system() == 'Windows':
		reduce_ram_usage(args['reduce-ram-usage'])
	
	#Base train on images
	setup_and_train_yolo(**args)
	webdl = WebDataLoader(args['total-maximum-images'], args['images-per-label'], args['max-train-images'], args['class-names'], args['input-dir'], PHOTO_DIRNAME)
	colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(args["class-names"]))]


	while webdl.has_next_batch():

		#Load the model
		model_fp = os.path.join( YOLO_DIR, 'runs', 'train', get_exp_dir(os.path.join(YOLO_DIR, 'runs', 'train')), 'weights', 'best.pt' )
		
		print(model_fp)

		batch_type = webdl.get_next_batch_type()
		
		img_batch, label_batch = webdl.get_next_batch() #Returns all image urls for the current batch into img_batch
		
		results = None #Initialize to none at beginning

		with torch.no_grad():
			
			#Try to move data to GPU if possible
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
			model = torch.hub.load('ultralytics/yolov5', 'custom', path = model_fp)

			model.to(device)
			model.eval()            
			results = model(img_batch) 

			#Clear up any hanging memory
			del model
			torch.cuda.empty_cache()
			gc.collect()


		num_images_taken = 0

		
		for i in range(len(img_batch)):


			add_to_dataset = False
			preds = results.pandas().xyxy[i].values.tolist()

			for pred in preds:
				if pred[4] >= args['min-conf-threshold'] and pred[-1] == label_batch[i]:        
					add_to_dataset = True
					break

			if add_to_dataset:

				img_fname = os.path.split(img_batch[i])[-1]
				img_ext = img_fname.split('.')[-1]


				#Copy the image from the photo_dir to the raw_dataset/train or raw_dataset/val directory
				train_val = 'train' if batch_type == 'valtrain' else 'valid'

				#Generate the label file in the raw_dataset/label/directory
				with open(os.path.join(YOLO_DIR, 'raw_dataset', train_val, 'labels', f'image-{webdl.get_total_ds_imgs() + 1}.txt'), 'w') as f: 
					
					img = Image.open(img_batch[i])
					img2 = cv2.imread(img_batch[i])


					w = int(img.size[0])
					h = int(img.size[1])
					if w == 0 or h == 0:
						add_to_dataset = False
						print(f'Error: {img_batch[i]} has width {w} and height {h}')
					else:
						for pred in preds:
							x, y, w, h = convert((w, h), (pred[0], pred[2], pred[1], pred[3]))
							print(f'{pred[5]} {x} {y} {w} {h}', file=f)
							
							plot_one_box(pred[0:4], img2, label=pred[-1], color=colors[pred[5]], line_thickness=3)


						shutil.copyfile(img_batch[i], os.path.join(YOLO_DIR, 'raw_dataset', train_val, 'images', f'image-{webdl.get_total_ds_imgs() + 1}.{img_ext}'))
						if args['save-bb-image'] and add_to_dataset:
							if train_val == 'train' and not os.path.exists(os.path.join(YOLO_DIR, 'raw_dataset', 'train', 'vis')):
								os.makedirs(os.path.join(YOLO_DIR, 'raw_dataset', 'train', 'vis'))
							elif train_val == 'valid' and not os.path.exists(os.path.join(YOLO_DIR, 'raw_dataset', 'valid', 'vis')):
								os.makedirs(os.path.join(YOLO_DIR, 'raw_dataset', 'valid', 'vis'))

							cv2.imwrite(os.path.join(YOLO_DIR, 'raw_dataset', train_val, 'vis', f'image-{webdl.get_total_ds_imgs() + 1}-vis.jpg'), img2)
						
						webdl.update_number_images_taken(1)

				#Clear up any hanging memory
				torch.cuda.empty_cache()
				gc.collect()

	


		#Update the total number of images on the webdl end
		webdl.update_number_images_taken(num_images_taken)

		#Train the model again with the updated dirs
		if batch_type == 'valtrain' and args['max-trains'] > 0:
			print('Training new model! More data, better model! :)')
			args['max-trains'] -= 1
			train_yolo(args['image-dimension'], args['train-batch'], args['train-epochs'])
			
			#Clear up any hanging memory
			torch.cuda.empty_cache()
			gc.collect()

	print('Training loop complete! :D')

	print('Moving all the files over to current directory ...')
	shutil.move(os.path.join(YOLO_DIR, 'raw_dataset'), '.')
	os.rename('raw_dataset', 'finalized_dataset')

	print('Deleting unncessary intermediate directories ...')
	
	#Perform cleanup 

	#1 - Delete the temp images we collected
	shutil.rmtree(PHOTO_DIRECTORY)
	if os.path.exists(PHOTO_DIRECTORY):
		os.rmdir(PHOTO_DIRECTORY)

	#2 - Clean up the scraper directory by removing extraneous json files
	[ os.remove(os.path.join('scraper', fname)) for fname in glob.glob('scraper/*.json') ]

	#3 - Move the best latest model run into the current for reference
	exp_dir = get_exp_dir(os.path.join(YOLO_DIR, 'runs', 'train'))
	model_dir = os.path.join( YOLO_DIR, 'runs', 'train', exp_dir)

	print('Moving the best model\'s directory into the current for reference')
	shutil.move(model_dir, '.')
	os.rename(exp_dir, 'best_model_info')

	print(f'Deleting the unnecessary {YOLO_DIR} directory')
	shutil.rmtree(YOLO_DIR)
	if os.path.exists(YOLO_DIR):
		os.rmdir(YOLO_DIR)


	

'''
Runtime Errors to note: 

RuntimeError: CUDA error: out of memory - Fix by lowering batch Size
RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED (re-init GPU, install 1.8.0)
'''