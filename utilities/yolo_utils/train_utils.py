import os
import shutil
from scraper.augmentations.run_augs import run_and_save_augments_on_image_sets
from utilities.dataset_setup.setup import setup_raw_dataset
import glob
from config import YOLO_URL, YOLO_DIR, PHOTO_DIRNAME, PHOTO_DIRECTORY

#DIM = 416, BATCH = 32, EPOCHS = 500, MODEL = 'yolov5s', WORKERS = 8)
def train_yolo(**args):
	os.system(f'python train.py --img {args['image-dimension']} --batch {args['train-batch']} --workers {WORKERS} --epochs {EPOCHS} --data {os.path.abspath("raw_dataset")}/data.yaml --weights {MODEL}.pt --cache' )

#, DIM, BATCH, EPOCHS, MAX_TRAIN_IMAGES , MODEL = 'yolov5s'
def setup_and_train_yolo(input_config_yaml):

	for pth in [YOLO_DIR, 'false', PHOTO_DIRNAME , PHOTO_DIRECTORY, 'finalized_dataset', 'best_model_info']:
		if os.path.exists(pth):
			shutil.rmtree(pth)
			if os.path.exists(pth):
				os.rmdir(pth)

	#TODO: add code to delete any cached json files
	[os.remove(file) for file in glob.glob("*.json" , recursive=True)]

	print('Cloning yolo repo ...')
	os.system(f"git clone {YOLO_URL}")

	print('Installing requirements ...')
	os.system(f"pip3 install -qr {YOLO_DIR}/requirements.txt")


	data_yaml_fp = setup_raw_dataset(input_config_yaml)


	global_cnt = 0
	for path in ['train', 'valid']:
		for pth in os.listdir(os.path.join(YOLO_DIR, 'raw_dataset', path, 'images')):
			os.rename(os.path.join(YOLO_DIR, 'raw_dataset' , path, 'images', pth), os.path.join(YOLO_DIR, 'raw_dataset' , path, 'images', f'image-{global_cnt + 1}.{pth.split(".")[-1]}'))
			os.rename(os.path.join(YOLO_DIR, 'raw_dataset', path, 'labels', f'{pth[:pth.rfind(".")]}.txt'), os.path.join(YOLO_DIR, 'raw_dataset' , path, 'labels', f'image-{global_cnt + 1}.txt'))
			global_cnt += 1


	#Generate image augmentations on the training images
	run_and_save_augments_on_image_sets(os.listdir(os.path.abspath(os.path.join(YOLO_DIR, 'raw_dataset', 'train', 'images'))), os.listdir(os.path.abspath(os.path.join(YOLO_DIR, 'raw_dataset', 'train', 'labels'))), MAX_TRAIN_IMAGES, os.path.abspath(os.path.join(YOLO_DIR, 'raw_dataset')), 'train')

	train_yolo(args['image-dimension'], args['train-batch'], args['train-epochs'], args['yolo-model'], args['train-workers'])