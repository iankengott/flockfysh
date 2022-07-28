import warnings
warnings.filterwarnings('ignore')

from urllib.parse import quote
import shutil
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager

import time
import json

BASE_URL = "https://images.search.yahoo.com/search/images;_ylt=Awr488YjaOBiVSwDNS2LuLkF;_ylc=X1MDOTYwNTc0ODMEX3IDMgRmcgMEZ3ByaWQDMDNHaEZ3c0VRUEMxZkxyYU9HdjlOQQRuX3N1Z2cDMTAEb3JpZ2luA2ltYWdlcy5zZWFyY2gueWFob28uY29tBHBvcwMwBHBxc3RyAwRwcXN0cmwDBHFzdHJsAzQEcXVlcnkDY29kZQR0X3N0bXADMTY1ODg3Mzg5Mw--?fr2=sb-top-images.search"


def gen_query_url(keywords, filters, extra_query_params=''):
    keywords_str = "&p=" + quote(keywords)
    query_url = BASE_URL + keywords_str
    return query_url


def image_url_from_webpage(driver, max_number=10000):
    image_urls = set()

    time.sleep(5)
    img_count = 0

    while True:
        image_elements = driver.find_elements_by_class_name("img")
        if len(image_elements) > max_number:
            break
        if len(image_elements) > img_count:
            img_count = len(image_elements)
            driver.execute_script(
                "window.scrollTo(0, document.body.scrollHeight);")
        else:
            smb = driver.find_elements_by_class_name("more-res")
            if len(smb) > 0 and smb[0].is_displayed():
                smb[0].click()
            else:
                break
        time.sleep(3)
    for image_element in image_elements:
        img = image_element.find_element_by_tag_name('img')
        src = img.get_attribute('src')
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
    print('\n\n')
    query_url = gen_query_url(keywords, filters, extra_query_params=extra_query_params)
    driver.set_window_size(3840, 2160)
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