import yaml
import os
from .job_config import TRAIN_SCRAPE_JOB, ANNOTATE_JOB


def get_train_scrape_params(job_name, input_params):

    #Load the default params first, and then pass them over as necessary
    default_params = None 
    with open(os.path.join(os.path.dirname(__file__),"default_params", "default_train_scrape.yaml"), 'r') as f:
        default_params = yaml.safe_load(f)
    
    if default_params == None:
        raise Exception(f'Unable to load {os.path.join(os.path.dirname(__file__),"default_params", "default_train_scrape.yaml")}')
         
    ret = {}
    for key in default_params.keys():
        if key in input_params: #Replace any default key_values with user-specified values
            ret[key] = input_params[key]
        else:
            if type(default_params[key]) == list and 'mandatory' in default_params[key][0] and default_params[key][0]['mandatory']:
                raise Exception(f'Parameter {key} is mandatory to specify for a {TRAIN_SCRAPE_JOB}, but isn\'t specified')
            elif type(default_params[key]) == list and 'mandatory' in default_params[key][0] and not default_params[key][0]['mandatory']:
                if 'value' not in default_params[key][0]:
                    raise Exception(f'Default train scraper YAML should have a \'value\' parameter accompanying any mandatory params.')
                ret[key] = default_params[key][0]['value']
            else:
                ret[key] = default_params[key]

    ret['job_name'] = job_name 
    ret['job_type'] = TRAIN_SCRAPE_JOB 
    #TODO: Check for conflicts in params / validate params , and handle them

    return ret

def get_annotate_params(job_name, input_params):
    #Load the default params first, and then pass them over as necessary
    default_params = None 

    #TODO: replace with annotate after it is operational
    with open(os.path.join(os.path.dirname(__file__),"default_params", "default_train_scrape.yaml"), 'r') as f:
        default_params = yaml.safe_load(f)
    
    if default_params == None:
        raise Exception(f'Unable to load {os.path.join(os.path.dirname(__file__),"default_params", "default_train_scrape.yaml")}')
         
    ret = {}
    for key, val in default_params:
        if key in input_params: #Replace any default key_values with user-specified values
            ret[key] = input_params[key]
        else:
            if 'mandatory' in val and val['mandatory']:
                raise Exception(f'Parameter {key} is mandatory to specify for a {ANNOTATE_JOB}, but isn\'t specified')
            elif 'mandatory' in val and not val['mandatory']:
                if 'value' not in val:
                    raise Exception(f'Default train scraper YAML should have a \'value\' parameter accompanying any mandatory params.')
                ret[key] = val['value']
            else:
                ret[key] = val

    ret['job_name'] = job_name 
    ret['job_type'] = ANNOTATE_JOB
    #TODO: Check for conflicts in params / validate params , and handle them
    return ret


