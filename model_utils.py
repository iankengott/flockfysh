import os
import shutil
import yaml
import sys
import time
import torch
from scraper import * 

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

yolo_url = 'https://github.com/ultralytics/yolov5'  #'https://github.com/WongKinYiu/yolov7' #Uncomment to use yolov7 
yolo_dir = yolo_url.split('/')[-1]

def setup_raw_dataset(yolo_dir, input_config_yaml):

    shutil.copytree(os.path.join(input_config_yaml["input_dir"], 'train'), os.path.join(yolo_dir, 'raw_dataset', 'train'))
    shutil.copytree(os.path.join(input_config_yaml["input_dir"], 'valid'), os.path.join(yolo_dir, 'raw_dataset', 'valid'))

    data_out_file = { "train" : f'../{yolo_dir}/raw_dataset/train/images', "val" : f"../{yolo_dir}/raw_dataset/valid/images", "nc" : len(input_config_yaml["class_names"]), "names" : input_config_yaml["class_names"]}

    with open(os.path.join(yolo_dir, 'raw_dataset', 'data.yaml'), 'w') as f:
        yaml.dump(data_out_file, f)

    return os.path.join(yolo_dir, 'raw_dataset', 'data.yaml')
    
def train_yolo(DIM = 416, BATCH = 32, EPOCHS = 500, MODEL = 'yolov5s6'):
    global yolo_dir

    os.chdir(yolo_dir)
    os.system(f'python3 train.py --img {DIM} --batch {BATCH} --epochs {EPOCHS} --data {os.path.abspath("raw_dataset")}/data.yaml --weights {MODEL}.pt --cache')
    os.chdir('../')

def setup_and_train_yolo(input_config_yaml, DIM, BATCH, EPOCHS, MODEL ):
    global yolo_dir
    global yolo_url

    for pth in [yolo_dir, 'false', 'photos', os.path.join('scraper', 'photos')]:
        if os.path.exists(pth):
            shutil.rmtree(pth)
            if os.path.exists(pth):
                os.rmdir(pth)

    print('Cloning yolo repo ...')
    os.system(f"git clone {yolo_url}")

    print('Installing requirements ...')
    os.system(f"pip install -qr yolov5/requirements.txt")


    data_yaml_fp = setup_raw_dataset(yolo_dir, input_config_yaml)
    train_yolo(DIM, BATCH, EPOCHS, MODEL)


#SRC: https://github.com/ultralytics/yolov5/blob/ea34f848a6afbe1fc0010745fdc5f356ed871909/utils/utils.py (lines 159-166)
def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y



def run_training_loop_object_detection_scrape(input_config_yaml, MAX_TOTAL_IMAGES = 7000, GOOD_THRESHOLD = 0.3, IMAGE_TRAIN_LIM = 5000, save_bb_image = True, DIM = 416, BATCH = 32, EPOCHS = 500, MODEL = 'yolov5s6'):
    
    global yolo_dir
    setup_and_train_yolo(input_config_yaml, DIM, BATCH, EPOCHS, MODEL)

    images_taken_so_far = 0
    imgs_to_take = 50
    debug_break = 2
    have_trained = True

    #Train yolo
    model_fp = os.path.join( yolo_dir, 'runs', 'train', 'exp' , 'weights', 'best.pt' )
    model = torch.hub.load('ultralytics/yolov5', 'custom', path = model_fp, force_reload = True)
    model.eval()


    print('Yolo Setup Complete')

    #Do the images duplicate if the code is re-run??
    while len(os.listdir(os.path.join(yolo_dir, 'raw_dataset', 'train', 'images'))) + len((os.listdir(os.path.join(yolo_dir, 'raw_dataset', 'valid', 'images')))) < MAX_TOTAL_IMAGES:
        #Scrape the images into the folder
        scrape_images(num_images_to_retrieve = images_taken_so_far + imgs_to_take, search_keys = input_config_yaml["class_names"] , save_dir = 'photos', no_chrome_gui = True, min_resolution = (0, 0), max_resolution = (9999, 9999), max_missed = 1000, num_workers = 1)

        if have_trained:

            exp_dir = f'exp{len(os.listdir(os.path.join(yolo_dir, "runs", "train", "exp")))}'
            model_fp = os.path.join( yolo_dir, 'runs', 'train', exp_dir if exp_dir != 'exp1' else 'exp', 'weights', 'best.pt' )
            model = torch.hub.load('ultralytics/yolov5', 'custom', path = model_fp, force_reload = True)
            model.eval()

        #Find list of good images (which model predicts score over > GOOD_THRESHOLD)
        good_images = []

        for classname in input_config_yaml["class_names"]:

            photo_urls = os.listdir(os.path.join('scraper', 'photos', classname))[images_taken_so_far:]
            results = model(photo_urls) 

            results.save(save_dir = os.path.join('scraper', 'photos', classname))

            for i in range(imgs_to_take):

                train_val = 'train' if len(os.listdir(os.path.join(yolo_dir, 'raw_datasets', 'train'))) <= IMAGE_TRAIN_LIM else 'valid'
                #We want to take the image only if it has the classlabel we are looking and the score over > GOOD_THRESHOLD
                preds = results.pandas().xyxy[i].values.tolist()
                for pred in preds:
                    if pred[4] >= GOOD_THRESHOLD and pred[-1] == classname:

                        shutil.copyfile(os.path.join('scraper', 'photos', classname, photo_urls[i]), os.path.join(yolo_dir, 'raw_dataset', train_val, 'images'))

                        if save_bb_image:
                            if train_val == 'train' and not os.path.exists(os.path.join(yolo_dir, 'raw_datasets', 'train', 'vis')):
                                os.makedirs(os.path.join(yolo_dir, 'raw_datasets', 'train', 'vis'))
                            elif train_val == 'valid' and not os.path.exists(os.path.join(yolo_dir, 'raw_datasets', 'valid', 'vis')):
                                os.makedirs(os.path.join(yolo_dir, 'raw_datasets', 'valid', 'vis'))
                            shutil.copyfile(os.path.join('scraper', 'photos', classname), os.path.join(yolo_dir, 'raw_datasets', train_val, 'vis', f'{photo_urls[i]}-bbox.jpg'))

                        xywh = (xyxy2xywh(pred[1:4]) / gn).view(-1).tolist()  # normalized xywh 
                        with open(os.path.join(yolo_dir, 'raw_dataset', train_val, 'labels', f'{photo_urls[i]}-label.txt'), 'w') as f: 
                            for pred2 in preds:
                                print(f'{pred2[5]} {xywh[0]} {xywh[1]} {xywh[2]} {xywh[3]}', file=f)

                        break


        images_taken_so_far += imgs_to_take
        imgs_to_take += imgs_to_take
        shutil.rmtree(os.path.join('scraper', 'photos'))

        have_trained = False
        if images_taken_so_far <= IMAGE_TRAIN_LIM:
            have_trained = True
            train_yolo()

        debug_break -= 1
        if debug_break == 0:
            break
    

    for pth in ['false', 'photos', os.path.join('scraper', 'photos')]:
        if os.path.exists(pth):
            shutil.rmtree(pth)
            if os.path.exists(pth):
                os.rmdir(pth)

    print('Training and annotation complete :)')