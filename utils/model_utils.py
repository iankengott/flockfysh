from cProfile import label
import enum
import os
import shutil
from cv2 import add, reduce
from matplotlib.pyplot import draw
import yaml
import sys
import time
import torch
import glob
from PIL import Image
import random
import cv2
import numpy as np
import re
import gc
import json
import platform

sys.path.append(os.path.join(os.path.dirname(__file__),'..' , 'scraper'))

from scraper.WebDataLoader import AnnotationDataLoader, WebDataLoader
from scraper.augmentations.run_augs import run_and_save_augments_on_image_sets


if sys.version_info[0] < 3:
	raise Exception("Must be using Python 3")


yolo_url = 'https://github.com/ultralytics/yolov5' 
yolo_dir = yolo_url.split('/')[-1]
PHOTO_DIRNAME = 'photos'
PHOTO_DIRECTORY = os.path.join('scraper', PHOTO_DIRNAME)

def setup_raw_dataset(yolo_dir, input_config_yaml):

	shutil.copytree(os.path.join(input_config_yaml["input_dir"], 'train'), os.path.join(yolo_dir, 'raw_dataset', 'train'))
	shutil.copytree(os.path.join(input_config_yaml["input_dir"], 'valid'), os.path.join(yolo_dir, 'raw_dataset', 'valid'))

	data_out_file = { "train" : f'../{yolo_dir}/raw_dataset/train/images', "val" : f"../{yolo_dir}/raw_dataset/valid/images", "nc" : len(input_config_yaml["class_names"]), "names" : input_config_yaml["class_names"]}

	with open(os.path.join(yolo_dir, 'raw_dataset', 'data.yaml'), 'w') as f:
		yaml.dump(data_out_file, f)

	return os.path.join(yolo_dir, 'raw_dataset', 'data.yaml')
	
def train_yolo(DIM = 416, BATCH = 32, EPOCHS = 500, MODEL = 'yolov5s', WORKERS = 8):
	global yolo_dir

	os.chdir(yolo_dir)

	print(MODEL)
	os.system(f'python train.py --img {DIM} --batch {BATCH} --workers {WORKERS} --epochs {EPOCHS} --data {os.path.abspath("raw_dataset")}/data.yaml --weights {MODEL}.pt --cache' )
	os.chdir('../')

  
  
def reduce_ram_usage(again):
	if not again:
		return
	
	venv_path = str(input('Enter the relative path to your virtual environment [ex: ..\\venv or \\venv]: '))
	dll_dir = os.path.abspath(os.path.join(venv_path, 'Lib\\site-packages\\torch\\lib'))

	if not os.path.exists(dll_dir):
		again = str(input('Path does not exist or torch is not installed. Try again? [Y/n]: ')).upper() == 'Y'
		reduce_ram_usage(again)
	else:
		dll_files = os.path.join(dll_dir, '*.dll')
		try:
			os.system(f'python ram_reducer.py --input="{dll_files}"')
		except:
			print('Reducing RAM usage process failed. Skipping reducing RAM usage process.')
	

def setup_and_train_yolo(input_config_yaml, DIM, BATCH, EPOCHS, MAX_TRAIN_IMAGES , MODEL = 'yolov5s'):
	global yolo_dir
	global yolo_url
	global PHOTO_DIRECTORY
	global PHOTO_DIRNAME

	for pth in [yolo_dir, 'false', PHOTO_DIRNAME , PHOTO_DIRECTORY, 'finalized_dataset', 'best_model_info']:
		if os.path.exists(pth):
			shutil.rmtree(pth)
			if os.path.exists(pth):
				os.rmdir(pth)

	#TODO: add code to delete any cached json files
	[os.remove(file) for file in glob.glob("*.json" , recursive=True)]

	print('Cloning yolo repo ...')
	os.system(f"git clone {yolo_url}")

	print('Installing requirements ...')
	os.system(f"pip3 install -qr {yolo_dir}/requirements.txt")

	data_yaml_fp = setup_raw_dataset(yolo_dir, input_config_yaml)

	global_cnt = 0
	for path in ['train', 'valid']:
		for pth in os.listdir(os.path.join(yolo_dir, 'raw_dataset', path, 'images')):
			os.rename(os.path.join(yolo_dir, 'raw_dataset' , path, 'images', pth), os.path.join(yolo_dir, 'raw_dataset' , path, 'images', f'image-{global_cnt + 1}.{pth.split(".")[-1]}'))
			os.rename(os.path.join(yolo_dir, 'raw_dataset', path, 'labels', f'{pth[:pth.rfind(".")]}.txt'), os.path.join(yolo_dir, 'raw_dataset' , path, 'labels', f'image-{global_cnt + 1}.txt'))
			global_cnt += 1

	#Generate image augmentations on the training images
	run_and_save_augments_on_image_sets(os.listdir(os.path.abspath(os.path.join(yolo_dir, 'raw_dataset', 'train', 'images'))), os.listdir(os.path.abspath(os.path.join(yolo_dir, 'raw_dataset', 'train', 'labels'))), MAX_TRAIN_IMAGES, os.path.abspath(os.path.join(yolo_dir, 'raw_dataset')), 'train')

	train_yolo(DIM, BATCH, EPOCHS, MODEL)

def convert(size, box):

	if size[0] == 0 or size[1] == 0:
		print(f'Error: width: {size[0]} and height: {size[1]}')
		print(box)

	dw = 1./size[0]
	dh = 1./size[1]
	x = (box[0] + box[1])/2.0
	y = (box[2] + box[3])/2.0
	w = box[1] - box[0]
	h = box[3] - box[2]
	x = x*dw
	w = w*dw
	y = y*dh
	h = h*dh
	return (x,y,w,h)

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
	# Plots one bounding box on image img
	tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
	color = color or [random.randint(0, 255) for _ in range(3)]
	c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
	cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
	if label:
		tf = max(tl - 1, 1)  # font thickness
		t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
		c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
		cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
		cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def get_exp_dir(exp_upper_dir):
	cur_num  = -1
	for folder in os.listdir(exp_upper_dir):
		if os.path.exists(os.path.join(exp_upper_dir, folder, 'weights', 'best.pt')):
			pot_num = (1 if folder == 'exp' else int(re.findall(r'\d+$', folder)[-1]))

			if pot_num > cur_num:
				cur_num = pot_num
	
	return f'exp{cur_num}' if cur_num != 1 else 'exp'


def run_training_object_detection_webscrape_loop(input_config_yaml, TOTAL_MAXIMUM_IMAGES = 7000, IMAGES_PER_LABEL = 500, MAX_TRAIN_IMAGES = 5000, CONFIDENCE_THRESHOLD = 0.3, SAVE_BB_IMAGE = True, DIM = 200, BATCH = 8, EPOCHS = 50, MAX_TRAINS = 3):
	global yolo_dir
	global PHOTO_DIRNAME
	global PHOTO_DIRECTORY

	#TODO: figure out a way to handle params without so much overflow
	MAX_TRAIN_IMAGES = IMAGES_PER_LABEL * len(input_config_yaml["class_names"])

	setup_and_train_yolo(input_config_yaml, DIM, 8, 100, MAX_TRAIN_IMAGES)
	webdl = WebDataLoader(TOTAL_MAXIMUM_IMAGES, IMAGES_PER_LABEL, MAX_TRAIN_IMAGES, input_config_yaml['class_names'], input_config_yaml['input_dir'], PHOTO_DIRNAME)
	colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(input_config_yaml["class_names"]))]

	while webdl.has_next_batch():

		#Load the model
		model_fp = os.path.join( yolo_dir, 'runs', 'train', get_exp_dir(os.path.join(yolo_dir, 'runs', 'train')), 'weights', 'best.pt' )
		
		batch_type = webdl.get_next_batch_type()
		
		img_batch, label_batch = webdl.get_next_batch() #Returns all image urls for the current batch into img_batch
		
		results = None #Initialize to none at beginning

		with torch.no_grad():
			
			#Try to move data to GPU if possible
			device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

			#TODO: Path issues like this are getting tedious and annoying. Shouldn't need to do relative paths for everything
			#TODO: We need some sort of a system that let's us shuffle global paths around efficiently
			model = torch.hub.load(os.path.join('..', yolo_dir),  'custom', source='local', path = model_fp)

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
				if pred[4] >= CONFIDENCE_THRESHOLD and pred[-1] == label_batch[i]:        
					add_to_dataset = True
					break

			if add_to_dataset:

				img_fname = os.path.split(img_batch[i])[-1]
				img_ext = img_fname.split('.')[-1]


				#Copy the image from the photo_dir to the raw_dataset/train or raw_dataset/val directory
				train_val = 'train' if batch_type == 'valtrain' else 'valid'

				#Generate the label file in the raw_dataset/label/directory
				with open(os.path.join(yolo_dir, 'raw_dataset', train_val, 'labels', f'image-{webdl.get_total_ds_imgs() + 1}.txt'), 'w') as f: 
					
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


						shutil.copyfile(img_batch[i], os.path.join(yolo_dir, 'raw_dataset', train_val, 'images', f'image-{webdl.get_total_ds_imgs() + 1}.{img_ext}'))
						if SAVE_BB_IMAGE and add_to_dataset:
							if train_val == 'train' and not os.path.exists(os.path.join(yolo_dir, 'raw_dataset', 'train', 'vis')):
								os.makedirs(os.path.join(yolo_dir, 'raw_dataset', 'train', 'vis'))
							elif train_val == 'valid' and not os.path.exists(os.path.join(yolo_dir, 'raw_dataset', 'valid', 'vis')):
								os.makedirs(os.path.join(yolo_dir, 'raw_dataset', 'valid', 'vis'))

							cv2.imwrite(os.path.join(yolo_dir, 'raw_dataset', train_val, 'vis', f'image-{webdl.get_total_ds_imgs() + 1}-vis.jpg'), img2)
						
						webdl.update_number_images_taken(1)

				#Clear up any hanging memory
				torch.cuda.empty_cache()
				gc.collect()

	


		#Update the total number of images on the webdl end
		webdl.update_number_images_taken(num_images_taken)

		#Train the model again with the updated dirs
		if batch_type == 'valtrain' and MAX_TRAINS > 0:
			print('Training new model! More data, better model! :)')
			MAX_TRAINS -= 1
			train_yolo(DIM, BATCH, EPOCHS)
			
			#Clear up any hanging memory
			torch.cuda.empty_cache()
			gc.collect()

	print('Training loop complete! :D')

	print('Moving all the files over to current directory ...')
	shutil.move(os.path.join(yolo_dir, 'raw_dataset'), '.')
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
	exp_dir = get_exp_dir(os.path.join(yolo_dir, 'runs', 'train'))
	model_dir = os.path.join( yolo_dir, 'runs', 'train', exp_dir)

	print('Moving the best model\'s directory into the current for reference')
	shutil.move(model_dir, '.')
	os.rename(exp_dir, 'best_model_info')

	print(f'Deleting the unnecessary {yolo_dir} directory')
	shutil.rmtree(yolo_dir)
	if os.path.exists(yolo_dir):
		os.rmdir(yolo_dir)


def run_object_detection_annotation(input_config_yaml, TOTAL_MAXIMUM_IMAGES = 7000, MAX_TRAIN_IMAGES = 5000, CONFIDENCE_THRESHOLD = 0.3, SAVE_BB_IMAGE = True, DIM = 200, BATCH = 16, EPOCHS = 50, MAX_TRAINS = 3):
	global yolo_dir
	global PHOTO_DIRNAME
	global PHOTO_DIRECTORY

	setup_and_train_yolo(input_config_yaml, DIM, 8, 100)
	
	#Homegenize all the data 
	adl = AnnotationDataLoader(input_config_yaml['class_names'] , input_config_yaml['input_dir'])
	colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(input_config_yaml["class_names"]))]

	#Store unclassified data 
	unlabeled_imgs = []
	unlabeled_labels = []

	while adl.has_next_batch():
		#Load the model
		model_fp = os.path.join( yolo_dir, 'runs', 'train', get_exp_dir(os.path.join(yolo_dir, 'runs', 'train')), 'weights', 'best.pt' )
		
		batch_type = adl.get_next_batch_type()
		
		img_batch, label_batch = adl.get_next_batch() #Returns all image urls for the current batch into img_batch
		
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
				if pred[4] >= CONFIDENCE_THRESHOLD and pred[-1] == label_batch[i]:        
					add_to_dataset = True
					break

			if add_to_dataset:

				img_fname = os.path.split(img_batch[i])[-1]
				img_name = img_fname.split('.')[0]

				#Copy the image from the photo_dir to the raw_dataset/train or raw_dataset/val directory
				train_val = 'train' if batch_type == 'valtrain' else 'valid'
				shutil.copyfile(img_batch[i], os.path.join(yolo_dir, 'raw_dataset', train_val, 'images', img_fname))

				#Generate the label file in the raw_dataset/label/directory
				with open(os.path.join(yolo_dir, 'raw_dataset', train_val, 'labels', f'{img_name}-label.txt'), 'w') as f: 
					
					img = Image.open(img_batch[i])
					img2 = cv2.imread(img_batch[i])


					w = int(img.size[0])
					h = int(img.size[1])

					if w == 0 or h == 0:
						add_to_dataset = False
						print(f'Error: {img_batch[i]} has width {w} and height {h}')

					if add_to_dataset:
						for pred in preds:
							x, y, w, h = convert((w, h), (pred[1], pred[3], pred[2], pred[4]))
							print(f'{pred[5]} {x} {y} {w} {h}', file=f)
							plot_one_box(pred[0:4], img2, label=pred[-1], color=colors[pred[5]], line_thickness=3)

				#Visualize image bounding boxes
				if SAVE_BB_IMAGE and add_to_dataset:
					if train_val == 'train' and not os.path.exists(os.path.join(yolo_dir, 'raw_dataset', 'train', 'vis')):
						os.makedirs(os.path.join(yolo_dir, 'raw_dataset', 'train', 'vis'))
					elif train_val == 'valid' and not os.path.exists(os.path.join(yolo_dir, 'raw_dataset', 'valid', 'vis')):
						os.makedirs(os.path.join(yolo_dir, 'raw_dataset', 'valid', 'vis'))

					cv2.imwrite(os.path.join(yolo_dir, 'raw_dataset', train_val, 'vis', f'{img_name}.jpg'), img2)
		
				#Clear up any hanging memory
				torch.cuda.empty_cache()
				gc.collect()

			else:
				unlabeled_imgs.append(img_batch[i])
				unlabeled_labels.append(label_batch[i])

		#Train the model again with the updated dirs
		if batch_type == 'valtrain' and MAX_TRAINS > 0:
			print('Training new model! More data, better model! :)')
			MAX_TRAINS -= 1
			train_yolo(DIM, BATCH, EPOCHS)
			
			#Clear up any hanging memory
			torch.cuda.empty_cache()
			gc.collect()

	#Create a new batch for the unbatched:
	adl.clear_batches()
	adl.set_data_and_batch_evenly(unlabeled_imgs, unlabeled_labels)

	while adl.has_next_batch():
	
		#Returns all image urls for the current batch into img_batch
		img_batch, label_batch = adl.get_next_batch() 
	
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

		for i in range(len(img_batch)):
			img_fname = os.path.split(img_batch[i])[-1]
			img_name = img_fname.split('.')[0]

			#Copy the image from the photo_dir to the raw_dataset/train or raw_dataset/val directory
			train_val = 'train' if batch_type == 'valtrain' else 'valid'
			shutil.copyfile(img_batch[i], os.path.join(yolo_dir, 'raw_dataset', train_val, 'images', img_fname))

			#Generate the label file in the raw_dataset/label/directory
			with open(os.path.join(yolo_dir, 'raw_dataset', train_val, 'labels', f'{img_name}-label.txt'), 'w') as f: 
				
				img = Image.open(img_batch[i])
				img2 = cv2.imread(img_batch[i])


				w = int(img.size[0])
				h = int(img.size[1])

				for pred in preds:
					x, y, w, h = convert((w, h), (pred[1], pred[3], pred[2], pred[4]))
					print(f'{pred[5]} {x} {y} {w} {h}', file=f)
					plot_one_box(pred[0:4], img2, label=pred[-1], color=colors[pred[5]], line_thickness=3)

			#Visualize image bounding boxes
			if SAVE_BB_IMAGE:
				if train_val == 'train' and not os.path.exists(os.path.join(yolo_dir, 'raw_dataset', 'train', 'vis')):
					os.makedirs(os.path.join(yolo_dir, 'raw_dataset', 'train', 'vis'))
				elif train_val == 'valid' and not os.path.exists(os.path.join(yolo_dir, 'raw_dataset', 'valid', 'vis')):
					os.makedirs(os.path.join(yolo_dir, 'raw_dataset', 'valid', 'vis'))

				cv2.imwrite(os.path.join(yolo_dir, 'raw_dataset', train_val, 'vis', f'{img_name}-vis.jpg'), img2)

			#Clear up any hanging memory
			torch.cuda.empty_cache()
			gc.collect()
	

	print('Moving all the files over to current directory ...')
	shutil.move(os.path.join(yolo_dir, 'raw_dataset'), '.')
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
	exp_dir = get_exp_dir(os.path.join(yolo_dir, 'runs', 'train'))
	model_dir = os.path.join( yolo_dir, 'runs', 'train', exp_dir)

	print('Moving the best model\'s directory into the current for reference')
	shutil.move(model_dir, '.')
	os.rename(exp_dir, 'best_model_info')

	print(f'Deleting the unnecessary {yolo_dir} directory')
	shutil.rmtree(yolo_dir)
	if os.path.exists(yolo_dir):
		os.rmdir(yolo_dir)

	
def generate_json_file(input_config_yaml, final_data_dir = 'finalized_dataset'):

	print('')
	out_dict = {
		"dataset_name": final_data_dir,
		"time_created": time.time(), 
		"labels": input_config_yaml["class_names"],
		"visualizations_added": os.path.exists(os.path.join(final_data_dir,'train' ,'vis')),
		"training_images": len(os.listdir(os.path.join(final_data_dir, 'train', 'images'))),
		"validation_images": len(os.listdir(os.path.join(final_data_dir, 'valid', 'images'))),
	}
 
	with open(os.path.join(final_data_dir, 'dataset-info.json'), 'w') as f:
		json.dump(out_dict, f,indent=6)

'''
Runtime Errors to note: 

RuntimeError: CUDA error: out of memory - Fix by lowering batch Size
RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED (re-init GPU, install 1.8.0)
'''