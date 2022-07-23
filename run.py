import yaml
import os
import sys
from model_utils import * 

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

sys.path.insert(1, './scraper')

def parse_params(input_yaml_file):
    if not os.path.exists(input_yaml_file):
        raise Exception(f'The input file {input_yaml_file} cannot be found in the current directory - {os.getcwd()}')
        
    with open(input_yaml_file, 'r') as f:
        params = yaml.safe_load(f)

        '''
        Run a couple of checks to make sure that 
        the YAML has a few required parameters 
        '''
        if not 'input_dir' in params:
            raise Exception(f'You need to specify an input directory by adding an input_dir attribute to the input YAML file {input_yaml_file}')

        if not 'class_names' in params:
            raise Exception(f'You need to specify a class names directory by adding a class_names attribute to the input YAML file {input_yaml_file}')

        if not os.path.exists(params['input_dir']):
            raise Exception(f'The input directory specified in the YAML - {params["input_dir"]} - doesn\'t exist in {os.getcwd()}')
        
        # for classnames in params['class_names']:
        #     if not os.path.exists(os.path.join(params["input_dir"], classnames)):
        #         raise Exception(f'Directory {os.path.join(params["input_dir"], classnames)} should exist, but do not.')

            # Uncomment at your peril ... it will force you to download 50 images per label folder
            # if len(os.listdir(os.path.join(params["input_dir"], classnames))) <= 50:
            #     raise Exception(f'The catgory {classnames}, with directory {os.path.join(params["input_dir"], classnames)}, has {len(os.listdir(os.path.join(params["input_dir"], classnames)))} images. It should have >= 30 images.')

    return params    


def get_input_file():
    if len(sys.argv) != 2:
        raise Exception(f'Unable to accept current params: {sys.argv[1:]}')

    input_yaml_file = sys.argv[1]

    if not input_yaml_file.endswith(input_yaml_file):
        print(f'Input file is NOT in YAML format. Please re-read input format')

    return input_yaml_file

def run():
    yaml_params = parse_params(get_input_file())
    run_training_loop_object_detection_scrape(yaml_params)

run()