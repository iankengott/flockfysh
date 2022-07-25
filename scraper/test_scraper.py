import sys
import os
from argparse import Namespace
from cygnusx1.bot import main as scrape_google_images
from bing_images import bing


def download_images_from_bing(classname, num_images):
    label_out_dir = os.path.join(OUTPUT_DIR, classname)
    if not os.path.exists(label_out_dir):
        os.makedirs(label_out_dir)
    
    bing.download_images(classname,
                        num_images,
                        output_dir=label_out_dir,
                        pool_size=10,
                        file_type="png",
                        force_replace=True,
                        extra_query_params='&first=1')

def download_images_from_google(classname, num_workers = 8):
    label_out_dir = os.path.join(OUTPUT_DIR, classname)
    if not os.path.exists(label_out_dir):
        os.makedirs(label_out_dir)

    args = Namespace(
        keywords = classname,
        workers = num_workers,
        headless = True, 
        use_suggestions = True,
        out_dir = label_out_dir,
    )

    scrape_google_images(args)

def download_by_chunk(classnames, MAX_IMAGES, ignore_excess = False):

    images_per_label = MAX_IMAGES // len(classnames)
    cur_image_count = []
    global OUTPUT_DIR #Change when integrating into webdataloader

    #Evenly split all the images first using the Bing Downloader
    for label in classnames:
        download_images_from_bing(label, images_per_label)
        cur_image_count.append(os.listdir(os.path.join(OUTPUT_DIR, label)))
    
    #Then distribute the remainder into each of the classnames
    for i in range(len(classnames)):
        
        #Make up for any shortages using the google downloader
        if cur_image_count[classnames] < images_per_label:
            download_images_from_google(classnames[i], num_workers = 8)

            if not ignore_excess:
                print('Excess has been specified to be removed, set ignore_excess to be True if the extra images is wanted')
                [os.remove(os.path.join(OUTPUT_DIR, classnames[i], f)) for f in os.listdir(os.path.join(OUTPUT_DIR, classnames[i]))[images_per_label:] ]

def batch_images(classnames, starting_img_per_batch = 50, MAX_IMAGES):
    img_batches = []
    label_batches = []

    #Precompute some information to increase efficiency
    completed = [False for i in range(len(classnames))]
    last_image_index = [0 for i in range(len(classnames))]
    total_files_per_class = [len(os.listdir(os.path.join("photos", classname))) for classname in classnames]
    abs_files = [os.walk(os.path.abspath(os.path.join("photos", classname))) for classname in classnames]

    #While we haven't exhausted all images
    while sum(completed) < len(classnames):

        image_batch = []
        label_batch = []                
        for i in range(len(classnames)):            
            if not completed[i]:
                image_batch.extend([ abs_files[last_image_index[i] : min(last_image_index[i] + starting_img_per_batch, total_files_per_class[i])] ])
                label_batch.extend([ classnames[i] for i in range([last_image_index[i] : min(last_image_index[i] + starting_img_per_batch, total_files_per_class[i])]) ])

                if last_image_index[i] + starting_img_per_batch > total_files_per_class[i]:
                    completed[i] = True

        img_batches.append(image_batch)
        label_batches.append(label_batch)
        starting_img_per_batch *= 2

    return (img_batches, label_batches)

