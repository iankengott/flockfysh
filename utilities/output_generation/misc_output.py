import os 
import time
import json
from config import BASE_DIR


def generate_json_file(input_config_yaml):

	final_data_dir = os.path.join(BASE_DIR, input_config_yaml['output-dataset-folder-name'])

	out_dict = {
		"dataset_name": final_data_dir,
		"time_finished_creation": time.time(), 
		"labels": input_config_yaml["class-names"],
		"visualizations_added": os.path.exists(os.path.join(final_data_dir,'train' ,'vis')),
		"training_images": len(os.listdir(os.path.join(final_data_dir, 'train', 'images'))),
		"validation_images": len(os.listdir(os.path.join(final_data_dir, 'valid', 'images'))),
	}
 
	with open(os.path.join(final_data_dir, 'dataset-info.json'), 'w') as f:
		json.dump(out_dict, f,indent=6)
