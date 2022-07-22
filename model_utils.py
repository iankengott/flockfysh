import os
import shutil
import yaml
import sys
import time
import torch

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

#SRC: https://github.com/ultralytics/yolov5/blob/ea34f848a6afbe1fc0010745fdc5f356ed871909/utils/utils.py (lines 159-166)
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y

def write_label(xyxy, file_path):
    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh 
    with open(file_path, 'a') as f: 
        f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format 

def inference_on_batch(model_fp, images, input_config_yaml, yolo_url='https://github.com/ultralytics/yolov5', IMAGE_TRAIN_LIM = 5000, save_bb_image = False):
    yolo_model = torch.load(model_fp)
    yolo_model.eval()

    results = yolo_model(images)

    yolo_dir = yolo_url.split('/')[-1]
    #Generate the labels files
    for i in range(len(results.xyxy)):
        train_val = 'train' if len(os.listdir(os.path.join(yolo_dir, 'raw_datasets', 'train'))) <= IMAGE_TRAIN_LIM else 'val'
        write_label(results.xyxy[i], os.path.join(yolo_dir, 'raw_datasets', train_val, 'labels' , f'{images[i][:images[i].split(".")[-1]]}.jpg' ) )

        if save_bb_image:
            if train_val == 'train' and not.os.path.exists(os.path.join(yolo_dir, 'raw_datasets', 'train', 'vis')):
                os.makedirs(os.path.join(yolo_dir, 'raw_datasets', 'train-vis'))
            elif train_val == 'val' and not.os.path.exists(os.path.join(yolo_dir, 'raw_datasets', 'val', 'vis'))
                os.makedirs(os.path.join(yolo_dir, 'raw_datasets', 'val-vis'))

            results.save(save_dir = os.path.join(yolo_dir, 'raw_datasets', train_val , 'vis', f'{images[i][:images[i].split(".")[-1]]}-bbox.jpg'))
        

    
