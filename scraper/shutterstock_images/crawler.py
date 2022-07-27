from urllib.parse import quote
import shutil
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from urllib.request import urlretrieve
from bs4 import BeautifulSoup

import time
import json

BASE_URL = "https://www.shutterstock.com/search/"


def gen_query_url(keywords, filters, extra_query_params='', page=1):
    print("Page " + str(page))
    keywords_str = quote(keywords)
    query_url = BASE_URL + keywords_str + f'?page={page}'
    return query_url


def image_url_from_webpage(driver, max_number=10000):
    image_urls = set()
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight)")
    time.sleep(4)
    data = driver.execute_script("return document.documentElement.outerHTML")
    scraper = BeautifulSoup(data, "lxml")
    img_container = scraper.find_all("img", {"class":"jss218"})
    for j in range(0, len(img_container)):
        img_src = img_container[j].get("src")
        image_urls.add(img_src)
    print(image_urls)
    return list(image_urls)
        


def crawl_image_urls(keywords, filters, max_number=10000, proxy=None, proxy_type="http", extra_query_params=''):
    chrome_path = shutil.which("chromedriver")
    chrome_path = "./bin/chromedriver" if chrome_path is None else chrome_path
    
    #Modified from original to make headless
    chrome_options = webdriver.ChromeOptions()
    chrome_options.headless = True 

    if proxy is not None and proxy_type is not None:
        chrome_options.add_argument(
            "--proxy-server={}://{}".format(proxy_type, proxy))
    
    #Update to handle webdriver
    driver = webdriver.Chrome(executable_path=ChromeDriverManager().install(), chrome_options=chrome_options)

    driver.set_window_size(1920, 1080)
    image_urls = list()
    i = 1
    while len(image_urls) < max_number:
        print(len(image_urls), max_number)
        query_url = gen_query_url(keywords, filters, extra_query_params=extra_query_params, page=i)
        driver.get(query_url)
        image_urls.extend(image_url_from_webpage(driver, max_number))
        i += 1

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