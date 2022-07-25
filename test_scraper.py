#Written to test the functionality of our scraper
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'scraper'))

from run import parse_params, get_input_file
from scraper.bing_images import bing
from scraper.WebDataLoader import WebDataLoader

input_config_yaml = parse_params(get_input_file())
webdl = WebDataLoader(2000, 1900, input_config_yaml['class_names'], input_config_yaml['input_dir'], 'photos')