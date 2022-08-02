import random 
import os 
from PIL import Image
import cv2
from scraper.DataLoaders.AnnotationDataLoader import AnnotationDataLoader


def run_object_detection_annotation(input_config_yaml, CONFIDENCE_THRESHOLD = 0.3, SAVE_BB_IMAGE = True, DIM = 200, BATCH = 16, EPOCHS = 50, MAX_TRAINS = 3):
	global yolo_dir
	global PHOTO_DIRNAME
	global PHOTO_DIRECTORY

	setup_and_train_yolo(input_config_yaml, DIM, 8, 100, len(os.listdir(os.path.join(yolo_dir, 'raw_dataset', 'train'))) * 5)
	
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
				img = Image.open(img_batch[i])
				img_ext = img_fname.split('.')[-1]

				w = int(img.size[0])
				h = int(img.size[1])
				if w == 0 or h == 0:
					add_to_dataset = False
					print(f'Error: {img_batch[i]} has width {w} and height {h}')


				train_val = 'train' if batch_type == 'valtrain' else 'valid'

				if train_val == 'valid':
					shutil.copyfile(img_batch[i], os.path.join(yolo_dir, 'raw_dataset', train_val, 'images', f'image-{adl.get_total_ds_imgs() + 1}.{img_ext}'))

					#Generate the label file in the raw_dataset/label/directory
					with open(os.path.join(yolo_dir, 'raw_dataset', train_val, 'labels', f'image-{adl.get_total_ds_imgs() + 1}.txt'), 'w') as f: 
						
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
						if train_val == 'valid' and not os.path.exists(os.path.join(yolo_dir, 'raw_dataset', 'valid', 'vis')):
							os.makedirs(os.path.join(yolo_dir, 'raw_dataset', 'valid', 'vis'))
						cv2.imwrite(os.path.join(yolo_dir, 'raw_dataset', train_val, 'vis', f'image-{adl.get_total_ds_imgs() + 1}-vis.jpg'), img2)
		
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
			train_val = 'train' if 'train' in img_batch[i] else 'valid'
			
			
			shutil.copyfile(img_batch[i], os.path.join(yolo_dir, 'raw_dataset', train_val, 'images', img_fname))

			with open(os.path.join(yolo_dir, 'raw_dataset', train_val, 'labels', f'image-{adl.get_total_ds_imgs() + 1}.txt'), 'w') as f: 
				
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
				if train_val == 'valid' and not os.path.exists(os.path.join(yolo_dir, 'raw_dataset', 'valid', 'vis')):
					os.makedirs(os.path.join(yolo_dir, 'raw_dataset', 'valid', 'vis'))
				cv2.imwrite(os.path.join(yolo_dir, 'raw_dataset', train_val, 'vis', f'image-{adl.get_total_ds_imgs() + 1}-vis.jpg'), img2)

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
