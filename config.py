import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_URL = 'https://github.com/ultralytics/yolov5' 
YOLO_DIR = YOLO_URL.split('/')[-1]
PHOTO_DIRNAME = 'photos'
PHOTO_DIRECTORY = os.path.abspath(os.path.join('scraper', PHOTO_DIRNAME))
