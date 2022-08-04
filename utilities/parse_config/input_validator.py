import yaml 
import os 
from .job_config import TRAIN_SCRAPE_JOB, ANNOTATE_JOB, DOWNLOAD_JOB
from .get_params_for_job import get_train_scrape_params, get_annotate_params, get_roboflow_params

def get_jobs_from_yaml_params(yaml_params):
    jobs = []

    #Assume all upper level keys are job names
    for key in yaml_params.keys():
        if not 'job-type' in yaml_params[key]:
            raise Exception(f'ERROR - The job with jobname {key} should have a subattribute \'job-type\' to tell us which type of job - {TRAIN_SCRAPE_JOB}, {ANNOTATE_JOB} - you would like done')

        if yaml_params[key]['job-type'] == TRAIN_SCRAPE_JOB:
            jobs.append(get_train_scrape_params(key, yaml_params[key]))
        elif yaml_params[key]['job-type'] == ANNOTATE_JOB:
            jobs.append(get_annotate_params(key, yaml_params[key]))
        elif yaml_params[key]['job-type'] == DOWNLOAD_JOB:
            if not 'api-name' in yaml_params[key]:
                raise Exception(f'ERROR - There should be an attribute \'api-name\' to tell us which type of api you would like us to use to download the dataset')
            if yaml_params[key]['api-name'] == 'roboflow':
                jobs.append(get_roboflow_params(key, yaml_params[key]))
        else:
            raise Exception(f'ERROR - The job with jobname {key} has an invalid job-type parameter: {yaml_params[key]["job-type"]} (valid ones are {TRAIN_SCRAPE_JOB} or {ANNOTATE_JOB})')
    
    return jobs
    

def get_input_yaml(input_yaml_file):
    if not os.path.exists(input_yaml_file):
        raise Exception(f'The input file {input_yaml_file} cannot be found in the current directory - {os.getcwd()}')
        
    #Validate the easy stuff as early
    if not input_yaml_file.endswith('.yaml'):
        raise Exception(f'The input file {input_yaml_file} in {os.getcwd()} isn\'t a YAML file (doesn\'t have the .yaml extension)')

    with open(input_yaml_file, 'r') as f:
        params = yaml.safe_load(f)

    return params    