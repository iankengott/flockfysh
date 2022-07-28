import warnings
warnings.filterwarnings('ignore')

from lib2to3.pgen2 import driver
from urllib.parse import quote
import shutil
from selenium import webdriver
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

import time
import json

from google_images.config import (GOOGLE_SUGGEST_CLASS, GOOGLE_THUBNAILS_XPATH,
                             GOOGLE_IMAGE_FULLSIZE_XPATH, GOOGLE_IMAGE_LOADING_BAR_XPATH)

BASE_URL = "https://www.google.com/search"


def gen_query_url(keywords, filters, extra_query_params=''):
    keywords_str = "?q=" + quote(keywords)
    query_url = BASE_URL + keywords_str + '&source=lnms&tbm=isch&safe=off'
    return query_url


def image_url_from_webpage(driver, max_number=10000):
    image_urls = set()

    time.sleep(5)

    try:
        last_height = driver.execute_script("return document.body.scrollHeight")
        reached_page_end = False
        while not reached_page_end:
            driver.execute_script(f"window.scrollTo(0, {last_height});")
            time.sleep(1)
            new_height = driver.execute_script("return document.body.scrollHeight")
            if last_height == new_height:
                reached_page_end = True
            else:
                last_height = new_height
            try:
                driver.find_element_by_class_name("mye4qd").click()
            except:
                continue
        image_elements = driver.find_elements(By.XPATH, GOOGLE_THUBNAILS_XPATH)
    except:
        pass

    for image_element in image_elements:
        src = image_element.get_attribute('src')
        image_urls.add(src)
    return list(image_urls)


def crawl_image_urls(keywords, filters, max_number=10000, proxy=None, proxy_type="http", extra_query_params=''):
    chrome_path = shutil.which("chromedriver")
    chrome_path = "./bin/chromedriver" if chrome_path is None else chrome_path
    
    #Modified from original to make headless
    chrome_options = webdriver.ChromeOptions()
    chrome_options.headless = True 
    chrome_options.add_argument('log-level=3')

    if proxy is not None and proxy_type is not None:
        chrome_options.add_argument(
            "--proxy-server={}://{}".format(proxy_type, proxy))
    
    #Update to handle webdriver
    driver = webdriver.Chrome(executable_path=ChromeDriverManager().install(), chrome_options=chrome_options)

    query_url = gen_query_url(keywords, filters, extra_query_params=extra_query_params)
    driver.get(query_url)
    image_urls = image_url_from_webpage(driver, max_number)
    driver.close()

    len_urls_unfilter = len(image_urls)

    if max_number > len(image_urls):
        output_num = len(image_urls)
    else:
        output_num = max_number

    print("Crawled {} image urls.".format(
        len(image_urls)))

    return image_urls[0:output_num], len_urls_unfilter


if __name__ == '__main__':
    images = crawl_image_urls(
        "cat png", "+filterui:aspect-square", max_number=10)
    for i in images:
        print(i+"\n")