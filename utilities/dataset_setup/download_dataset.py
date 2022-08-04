from roboflow import Roboflow
import os
from config import BASE_DIR

def download_to_local_directory_robo(api_info):
    print(api_info)    
    rf = Roboflow(api_key = api_info['api-key'])
    project = rf.workspace(api_info['workspace-name']).project(api_info['project-name'])
    dataset = project.version(api_info['project-version']).download("yolov5", location = os.path.join(BASE_DIR, api_info['output-dirname']))


def download_dataset(dataset_type, api_info):
    if dataset_type == 'roboflow':
        download_to_local_directory_robo(api_info)