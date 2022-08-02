import shutil
import os
import yaml 

def setup_raw_dataset(yolo_dir, input_config_yaml):

	shutil.copytree(os.path.join(input_config_yaml["input_dir"], 'train'), os.path.join(yolo_dir, 'raw_dataset', 'train'))
	shutil.copytree(os.path.join(input_config_yaml["input_dir"], 'valid'), os.path.join(yolo_dir, 'raw_dataset', 'valid'))

	data_out_file = { "train" : f'../{yolo_dir}/raw_dataset/train/images', "val" : f"../{yolo_dir}/raw_dataset/valid/images", "nc" : len(input_config_yaml["class_names"]), "names" : input_config_yaml["class_names"]}

	with open(os.path.join(yolo_dir, 'raw_dataset', 'data.yaml'), 'w') as f:
		yaml.dump(data_out_file, f)

	return os.path.join(yolo_dir, 'raw_dataset', 'data.yaml')
