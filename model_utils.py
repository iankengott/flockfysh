import os
import shutil
import yaml
import sys
import time

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

def setup_raw_dataset(yolo_dir, input_config_yaml):

    shutil.copytree(os.path.join(input_config_yaml["input_dir"], 'train'), os.path.join(yolo_dir, 'raw_dataset', 'train'))
    shutil.copytree(os.path.join(input_config_yaml["input_dir"], 'valid'), os.path.join(yolo_dir, 'raw_dataset', 'valid'))

    data_out_file = { "train" : f'../{yolo_dir}/raw_dataset/train/images', "val" : f"../{yolo_dir}/raw_dataset/valid/images", "nc" : len(input_config_yaml["class_names"]), "names" : input_config_yaml["class_names"]}

    with open(os.path.join(yolo_dir, 'raw_dataset', 'data.yaml'), 'w') as f:
        yaml.dump(data_out_file, f)

    return os.path.join(yolo_dir, 'raw_dataset', 'data.yaml')
    

def setup_and_train_yolo(input_config_yaml, yolo_url='https://github.com/ultralytics/yolov5', DIM = 416, BATCH = 8, EPOCHS = 500, MODEL = 'yolov5s6' ):

    yolo_dir = yolo_url.split('/')[-1]

    if os.path.exists(yolo_dir):
        shutil.rmtree(yolo_dir)
    
    print('Cloning yolo repo ...')
    os.system(f"git clone {yolo_url}")


    print('Installing requirements ...')
    os.system(f"pip install -qr yolov5/requirements.txt")


    data_yaml_fp = setup_raw_dataset(yolo_dir, input_config_yaml)

    os.chdir(yolo_dir)
    os.system(f'python3 train.py --img {DIM} --batch {BATCH} --epochs {EPOCHS} --data {os.path.abspath(os.path.join("raw_dataset"))}/data.yaml --weights {MODEL}.pt --cache')

