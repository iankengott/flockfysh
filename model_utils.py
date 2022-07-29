import os
import shutil
from matplotlib.pyplot import draw
import yaml
import sys
import time
import torch
import glob
from PIL import Image
import random
import cv2
import numpy as np
import re
import gc

sys.path.append(os.path.join(os.path.dirname(__file__), 'scraper'))

from scraper.WebDataLoader import WebDataLoader


if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")


yolo_url = 'https://github.com/WongKinYiu/yolov7'  # 'https://github.com/ultralytics/yolov5' #Uncomment to use yolov7 
yolo_dir = yolo_url.split('/')[-1]
PHOTO_DIRNAME = 'photos'
PHOTO_DIRECTORY = os.path.join('scraper', PHOTO_DIRNAME)

def setup_raw_dataset(yolo_dir, input_config_yaml):

    shutil.copytree(os.path.join(input_config_yaml["input_dir"], 'train'), os.path.join(yolo_dir, 'raw_dataset', 'train'))
    shutil.copytree(os.path.join(input_config_yaml["input_dir"], 'valid'), os.path.join(yolo_dir, 'raw_dataset', 'valid'))

    data_out_file = { "train" : f'../{yolo_dir}/raw_dataset/train/images', "val" : f"../{yolo_dir}/raw_dataset/valid/images", "nc" : len(input_config_yaml["class_names"]), "names" : input_config_yaml["class_names"]}

    with open(os.path.join(yolo_dir, 'raw_dataset', 'data.yaml'), 'w') as f:
        yaml.dump(data_out_file, f)

    return os.path.join(yolo_dir, 'raw_dataset', 'data.yaml')
    
def train_yolo(DIM = 416, BATCH = 32, EPOCHS = 500, MODEL = 'yolov7', WORKERS = 8):
    global yolo_dir

    os.chdir(yolo_dir)
    os.system(f'python train.py --workers {WORKERS} --device 0 --epochs {EPOCHS} --batch-size {BATCH} --data {os.path.abspath(os.path.join("raw_dataset", "data.yaml"))} --img {DIM} {DIM} --cfg cfg/training/yolov7.yaml --weights "" --name {MODEL} --hyp data/hyp.scratch.p5.yaml')
#    os.system(f'python3 train.py --img {DIM} --batch {BATCH} --epochs {EPOCHS} --data {os.path.abspath("raw_dataset")}/data.yaml --weights {MODEL}.pt --cache')
    os.chdir('../')

def setup_and_train_yolo(input_config_yaml, DIM, BATCH, EPOCHS, MODEL = 'yolov7'):
    global yolo_dir
    global yolo_url
    global PHOTO_DIRECTORY
    global PHOTO_DIRNAME

    for pth in [yolo_dir, 'false', PHOTO_DIRNAME , PHOTO_DIRECTORY]:
        if os.path.exists(pth):
            shutil.rmtree(pth)
            if os.path.exists(pth):
                os.rmdir(pth)

    print('Reducing RAM usage ...')
    ram = str(input('Would you like RAM usage to be decreased? [Y/n]: ')).upper()
    if ram == 'Y':
        again = True
        while again:
            venv_path = str(input('Enter the relative path to your virtual environment [ex: ..\\venv or \\venv]: '))
            dll_dir = os.path.abspath(os.path.join(venv_path, 'Lib\\site-packages\\torch\\lib'))
            print(dll_dir)
            if not os.path.exists(dll_dir):
                again = str(input('Path does not exist or torch is not installed. Try again? [Y/n]: ')).upper() == 'Y'
                continue
            again = False
            dll_files = os.path.join(dll_dir, '*.dll')
            try:
                os.system(f'python ram_reducer.py --input="{dll_files}"')
            except:
                print('Reducing RAM usage process failed. Skipping reducing RAM usage process.')

    print('Cloning yolo repo ...')
    os.system(f"git clone {yolo_url}")

    print('Installing requirements ...')
    os.system(f"pip3 install -qr {yolo_dir}/requirements.txt")

    data_yaml_fp = setup_raw_dataset(yolo_dir, input_config_yaml)
    train_yolo(DIM, BATCH, EPOCHS, MODEL)

def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh
    return (x,y,w,h)

def draw_bbox(image, xmin, ymin, xmax, ymax, text=None):
    
    """
    This functions draws one bounding box on an image.
    
    Input: Image (numpy array)
    Output: Image with the bounding box drawn in. (numpy array)
    
    If there are multiple bounding boxes to draw then simply
    run this function multiple times on the same image.
    
    Set text=None to only draw a bbox without
    any text or text background.
    E.g. set text='Balloon' to write a 
    title above the bbox.
    
    xmin, ymin --> coords of the top left corner.
    xmax, ymax --> coords of the bottom right corner.
    
    """


    w = xmax - xmin
    h = ymax - ymin

    # Draw the bounding box
    # ......................
    
    start_point = (xmin, ymin) 
    end_point = (xmax, ymax) 
    bbox_color = (255, 0, 0) 
    bbox_thickness = 15

    image = cv2.rectangle(image, start_point, end_point, bbox_color, bbox_thickness) 
    
    
    
    # Draw the tbackground behind the text and the text
    # .................................................
    
    # Only do this if text is not None.
    if text:
        
        # Draw the background behind the text
        text_bground_color = (0,0,0) # black
        cv2.rectangle(image, (xmin, ymin-150), (xmin+w, ymin), text_bground_color, -1)

        # Draw the text
        text_color = (255, 255, 255) # white
        font = cv2.FONT_HERSHEY_DUPLEX
        origin = (xmin, ymin-30)
        fontScale = 3
        thickness = 10

        image = cv2.putText(image, text, origin, font, 
                           fontScale, text_color, thickness, cv2.LINE_AA)



    return image

def get_exp_dir(exp_upper_dir):
    cur_num  = -1
    for folder in os.listdir(exp_upper_dir):
        if os.path.exists(os.path.join(exp_upper_dir, folder, 'weights', 'best.pt')):
            pot_num = (1 if folder == 'exp' else int(re.findall(r'\d+$', folder)[-1]))

            if pot_num > cur_num:
                cur_num = pot_num
    
    return f'exp{cur_num}' if cur_num != 1 else 'exp'
            

def run_training_object_detection_webscrape_loop(input_config_yaml, TOTAL_MAXIMUM_IMAGES = 7000, MAX_TRAIN_IMAGES = 5000, CONFIDENCE_THRESHOLD = 0.3, SAVE_BB_IMAGE = True, DIM = 200, BATCH = 16, EPOCHS = 50, MAX_TRAINS = 3):
    global yolo_dir
    global PHOTO_DIRNAME
    global PHOTO_DIRECTORY

    setup_and_train_yolo(input_config_yaml, DIM, 32, 100)
    webdl = WebDataLoader(TOTAL_MAXIMUM_IMAGES, MAX_TRAIN_IMAGES, input_config_yaml['class_names'], input_config_yaml['input_dir'], PHOTO_DIRNAME)

    while webdl.has_next_batch():

        #Load the model
        model_fp = os.path.join( yolo_dir, 'runs', 'train', get_exp_dir(os.path.join(yolo_dir, 'runs', 'train')), 'weights', 'best.pt' )
        
        batch_type = webdl.get_next_batch_type()
        
        img_batch, label_batch = webdl.get_next_batch() #Returns all image urls for the current batch into img_batch
        
        results = None #Initialize to none at beginning

        with torch.no_grad():
            
            #Try to move data to GPU if possible
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = torch.hub.load('WongKinYiu/yolov7', 'yolov7')
            model.load_state_dict(torch.load(model_fp)['model'].state_dict())

            model.to(device)
            model.eval()            
            results = model(img_batch) 

            #Clear up any hanging memory
            del model
            torch.cuda.empty_cache()
            gc.collect()


        num_images_taken = 0
        
        for i in range(len(img_batch)):


            add_to_dataset = False
            preds = results.pandas().xyxy[i].values.tolist()
            del results

            for pred in preds:
                if pred[4] >= CONFIDENCE_THRESHOLD and pred[-1] == label_batch[i]:        
                    add_to_dataset = True
                    break

            if add_to_dataset:

                img_fname = os.path.split(img_batch[i])[-1]
                img_name = img_fname.split('.')[0]

                #Copy the image from the photo_dir to the raw_dataset/train or raw_dataset/val directory
                train_val = 'train' if batch_type == 'valtrain' else 'valid'
                shutil.copyfile(img_batch[i], os.path.join(yolo_dir, 'raw_dataset', train_val, 'images', img_fname))

                #Generate the label file in the raw_dataset/label/directory
                with open(os.path.join(yolo_dir, 'raw_dataset', train_val, 'labels', f'{img_name}-label.txt'), 'w') as f: 
                    
                    img = Image.open(img_batch[i])
                    img2 = cv2.imread(img_batch[i])


                    w = int(img.size[0])
                    h = int(img.size[1])

                    for pred in preds:
                        x, y, w, h = convert((w, h), (pred[1], pred[3], pred[2], pred[4]))
                        print(f'{pred[5]} {x} {y} {w} {h}', file=f)
                        img2 = draw_bbox(img2, int(pred[1]), int(pred[2]), int(pred[3]), int(pred[4]), text= pred[-1])

                #Visualize image bounding boxes
                if SAVE_BB_IMAGE:
                    if train_val == 'train' and not os.path.exists(os.path.join(yolo_dir, 'raw_dataset', 'train', 'vis')):
                        os.makedirs(os.path.join(yolo_dir, 'raw_dataset', 'train', 'vis'))
                    elif train_val == 'valid' and not os.path.exists(os.path.join(yolo_dir, 'raw_dataset', 'valid', 'vis')):
                        os.makedirs(os.path.join(yolo_dir, 'raw_dataset', 'valid', 'vis'))

                    cv2.imwrite(os.path.join(yolo_dir, 'raw_dataset', train_val, 'vis', f'{img_name}-vis.jpg'), img2)
        
                #Clear up any hanging memory
                torch.cuda.empty_cache()
                gc.collect()

                num_images_taken += 1
    


        #Update the total number of images on the webdl end
        webdl.update_number_images_taken(num_images_taken)

        #Train the model again with the updated dirs
        if batch_type == 'valtrain' and MAX_TRAINS > 0:
            print('Training new model! More data, better model! :)')
            MAX_TRAINS -= 1
            train_yolo(DIM, BATCH, EPOCHS)
            
            #Clear up any hanging memory
            torch.cuda.empty_cache()
            gc.collect()

    print('Training loop complete! :D')

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

def run_object_detection_annotation(input_config_yaml, TOTAL_MAXIMUM_IMAGES = 7000, MAX_TRAIN_IMAGES = 5000, CONFIDENCE_THRESHOLD = 0.3, SAVE_BB_IMAGE = True, DIM = 200, BATCH = 16, EPOCHS = 50, MAX_TRAINS = 3):
    global yolo_dir
    global PHOTO_DIRNAME
    global PHOTO_DIRECTORY

    setup_and_train_yolo(input_config_yaml, DIM, 32, 100)
    

    
    pass

'''
Runtime Errors to note: 

RuntimeError: CUDA error: out of memory - Fix by lowering batch Size
RuntimeError: cuDNN error: CUDNN_STATUS_NOT_INITIALIZED (re-init GPU, install 1.8.0)
'''