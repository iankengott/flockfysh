import yaml
import os
import sys
from utilities.pipelines.training_webscrape_loop import run_training_object_detection_webscrape_loop
from utilities.parse_config.input_validator import get_jobs_from_yaml_params, get_input_yaml 
from utilities.parse_config.job_config import TRAIN_SCRAPE_JOB, ANNOTATE_JOB, DOWNLOAD_JOB, SUPPORTED_DOWNLOAD_APIS
from utilities.output_generation.misc_output import generate_json_file
from utilities.dataset_setup.download_dataset import download_dataset

if sys.version_info[0] < 3:
    raise Exception("Must be using Python 3")

def run():
    #Load the YAML params first
    yaml_params = get_input_yaml(sys.argv[1])
    jobs = get_jobs_from_yaml_params(yaml_params)

    for job in jobs:
        if job['job_type'] == TRAIN_SCRAPE_JOB:
            print(f'Running train scrape job with name {job["job_name"]}')
            run_training_object_detection_webscrape_loop(**job)
            generate_json_file(job)
        
        elif job['job_type'] == ANNOTATE_JOB:
            print(f'Running annotate job with name {job["job_name"]}')

        elif job['job_type'] == DOWNLOAD_JOB:
            download_dataset(job['api-name'], job)
        


if __name__ == '__main__':
    run()
