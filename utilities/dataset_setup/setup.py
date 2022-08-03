import shutil
import os
import yaml 
from config import  YOLO_DIR

def setup_raw_dataset(input_config_yaml):

	shutil.copytree(os.path.join(input_config_yaml["input-dir"], 'train'), os.path.join(YOLO_DIR, 'raw_dataset', 'train'))
	shutil.copytree(os.path.join(input_config_yaml["input-dir"], 'valid'), os.path.join(YOLO_DIR, 'raw_dataset', 'valid'))

	data_out_file = { "train" : f'../{YOLO_DIR}/raw_dataset/train/images', "val" : f"../{YOLO_DIR}/raw_dataset/valid/images", "nc" : len(input_config_yaml["class-names"]), "names" : input_config_yaml["class-names"]}

	with open(os.path.join(YOLO_DIR, 'raw_dataset', 'data.yaml'), 'w') as f:
		yaml.dump(data_out_file, f)

	return os.path.join(YOLO_DIR, 'raw_dataset', 'data.yaml')
