# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 11:02:06 2020

@author: OHyic

"""
#Import libraries
import os
import concurrent.futures
from GoogleImageScraper import GoogleImageScraper
from patch import webdriver_executable

def scrape_images(num_images_to_retrieve = 50, search_keys = ["cat"] , save_dir = './photos', no_chrome_gui = True, min_resolution = (0, 0), max_resolution = (9999, 9999), max_missed = 1000, num_workers = 1):
    #Define file path
    webdriver_path = os.path.normpath(os.path.join(os.getcwd(), 'webdriver', webdriver_executable()))

    for search_key in search_keys:
        image_scraper = GoogleImageScraper(
        webdriver_path, save_dir, search_key, num_images_to_retrieve, no_chrome_gui, min_resolution, max_resolution)
        image_urls = image_scraper.find_image_urls()
        image_scraper.save_images(image_urls)

scrape_images(num_images_to_retrieve = 10)