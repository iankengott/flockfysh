#Written to test the functionality of our scraper
import sys
import os

sys.path.append(os.path.abspath(os.path.join('scraper')))

print(sys.path)

from scraper.WebDataLoader import WebDataLoader

#Toggle these variables to test scraping
TOTAL_MAXIMUM_IMAGES = 2000
IMAGES_PER_LABEL = 500 #TOTAL MAXIMIM IMAGES = IMAGES PER LABEL * (number of labels). If TOTAL MAXIMUM IMAGES != [IMAGES PER LABEL * (number of labels)], IT IS OVERRIDDEN WITH IMAGES PER LABEL * (number of labels)
MAX_TRAIN_IMAGES = 1900


webdl = WebDataLoader(TOTAL_MAXIMUM_IMAGES, IMAGES_PER_LABEL, MAX_TRAIN_IMAGES, ['Bed Bug', 'Fire ant', 'Tick', 'Wasp'], os.path.abspath(os.path.join('robo')), os.path.abspath(os.path.join('scraper', 'photos')))
