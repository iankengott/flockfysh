from scraper import * 
from utilities import *
from . import config 
import sys


from .config import BASE_DIR
import os 

sys.path.append(os.path.abspath(os.path.join(BASE_DIR, 'yolov5')))
sys.path.append(os.path.abspath(os.path.join(BASE_DIR, 'scraper')))

name = "flockfysh"