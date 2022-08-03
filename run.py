import yaml
import os
import sys
from utilities.pipelines.training_webscrape_loop import run_training_object_detection_webscrape_loop
from utilities.parse_config.input_validator import get_jobs_from_yaml_params, get_input_yaml 
from utilities.parse_config.job_config import TRAIN_SCRAPE_JOB, ANNOTATE_JOB
from utilities.output_generation.misc_output import generate_json_file

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
        
        elif job['job_type'] == ANNOTATE_JOB:
            print(f'Running annotate job with name {job["job_name"]}')
        


    run_training_object_detection_webscrape_loop(yaml_params, TOTAL_MAXIMUM_IMAGES=2000, MAX_TRAIN_IMAGES=1900)
    generate_json_file(yaml_params)

if __name__ == '__main__':
    run()
