import warnings
warnings.filterwarnings('ignore')

from urllib.parse import quote
import shutil
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys


import time
import json

BASE_URL = "https://www.bing.com/images/search?"


def gen_query_url(keywords, filters, extra_query_params=''):
    keywords_str = "&q=" + quote(keywords)
    query_url = BASE_URL + keywords_str
    if len(filters) > 0:
        query_url += "&qft="+filters
    query_url += extra_query_params
    return query_url


def image_url_from_webpage(driver, max_number=10000):
    image_urls = set()

    time.sleep(5)
    img_count = 0

    lookforbutton = True
    reached_page_end = False
    last_height = driver.execute_script("return document.body.scrollHeight")

    while not reached_page_end:
        image_elements = driver.find_elements_by_class_name("iusc")
        if len(image_elements) > max_number:
            break
        if lookforbutton:
            driver.execute_script(
                "window.scrollBy(0, 1000);")
            smb = driver.find_elements_by_class_name("btn_seemore")
            if len(smb) > 0 and smb[0].is_displayed():
                print('See more button clicked.')
                smb[0].click()
                lookforbutton = False
        else:
            driver.execute_script("window.scrollBy(0, 1000);")
            new_height = driver.execute_script("return document.body.scrollHeight")
            if last_height == new_height:
                reached_page_end = True
            else:
                last_height = new_height

        time.sleep(1)
    for image_element in image_elements:
        m_json_str = image_element.get_attribute("m")
        m_json = json.loads(m_json_str)
        image_urls.add(m_json["murl"])
    return list(image_urls)


def crawl_image_urls(keywords, filters, max_number=10000, proxy=None, proxy_type="http", extra_query_params=''):
    chrome_path = shutil.which("chromedriver")
    chrome_path = "./bin/chromedriver" if chrome_path is None else chrome_path
    
    #Modified from original to make headless
    chrome_options = webdriver.ChromeOptions()
    chrome_options.headless = False 

    if proxy is not None and proxy_type is not None:
        chrome_options.add_argument(
            "--proxy-server={}://{}".format(proxy_type, proxy))
    
    #Update to handle webdriver
    driver = webdriver.Chrome(executable_path=ChromeDriverManager().install(), chrome_options=chrome_options)

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