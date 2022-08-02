import os 
import json

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
