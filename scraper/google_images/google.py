try:
    from crawler import crawl_image_urls
except ImportError:  # Python 3
    from .crawler import crawl_image_urls
from typing import List
from multiprocessing.pool import ThreadPool
from time import time as timer
import os
import math

import sys
sys.path.append('../scraper')
try:
    from scraper.util import get_file_name, rename, make_image_dir, download_image
except ImportError:  # Python 3
    from scraper.util import get_file_name, rename, make_image_dir, download_image

_FINISH = False


def fetch_image_urls(
    query: str,
    limit: int = 200,
    file_type: str = '',
    filters: str = '',
    extra_query_params: str =''
) -> List[str]:
    result = list()
    keywords = query
    if len(file_type) > 0:
        keywords = query + " " + file_type
    urls, len_urls_unfilter = crawl_image_urls(keywords, filters, limit, extra_query_params=extra_query_params)
    for url in urls:
        if isValidURL(url, file_type) and url not in result:
            result.append(url)
            if len(result) >= limit:
                break
    return result, len_urls_unfilter


def isValidURL(url, file_type):
    if len(file_type) < 1:
        return True
    return url.endswith(file_type)


def download_images(
    query: str,
    limit: int = 20,
    output_dir='',
    pool_size: int = 20,
    file_type: str = '',
    filters: str = '',
    force_replace=False,
    extra_query_params: str =''
):
    start = timer()
    image_dir = make_image_dir(output_dir, force_replace)

    # Fetch more image URLs to avoid some images are invalid.
    max_number = math.ceil(limit)
    urls, len_urls_unfilter = fetch_image_urls(query, max_number, file_type, filters, extra_query_params=extra_query_params)
    entries = get_image_entries(urls, image_dir)

    print("Downloading images.")
    ps = pool_size
    if limit < pool_size:
        ps = limit
    
    cts = download_image_entries(entries, ps, limit)

    print("Images downloaded.")
    elapsed = timer() - start
    print("Elapsed time: %.2fs\n\n" % elapsed)
    return cts, len_urls_unfilter

def download_image_entries(entries, pool_size, limit):
    global _FINISH
    counter = 1
    _FINISH = False
    pool = ThreadPool(pool_size)
    results = pool.imap_unordered(
        download_image_with_thread, entries)
    for (url, result) in results:
        if counter > limit:
            _FINISH = True
            pool.terminate()
            break
        if result:
            print("#{} {} Downloaded".format(counter, url))
            counter = counter + 1
    
    return counter


def get_image_entries(urls, dir):
    entries = []
    i = 0
    for url in urls:
        path = dir
        entries.append((url, path))
        i = i + 1
    return entries


def download_image_with_thread(entry):
    if _FINISH:
        return
    url, path = entry
    result = download_image(url, path)
    return (url, result)


if __name__ == '__main__':
    download_images("cat",
                    20,
                    output_dir="/Users/catchzeng/Desktop/cat",
                    pool_size=10,
                    file_type="png",
                    force_replace=True,
                    extra_query_params='&first=100')