from matplotlib import image
from PIL import Image
import os
import numpy as np

from .augs import get_augmentations



def run_augments_on_image(img_name, bboxes, max_images_to_generate = 500):
    
    ret = []
    img = np.array(Image.open(img_name), dtype=np.uint8)
    transforms = get_augmentations()

    for i in range(min(len(transforms), max_images_to_generate)):
        transformed = transforms[i](image=img, bboxes = bboxes)
        ret.append((transformed["image"] , transformed["bboxes"]))

    return ret

def run_and_save_augments_on_image_sets(batch_img_names, bboxes_urls, max_images_to_generate, dataset_dir, trainval):
    
    print('Perfoming Data Augmentations on Training Images to increase training size. ')
    

    num_images = 0
    for i in range(len(batch_img_names)):

        bboxes = []
        with open(os.path.join(dataset_dir, trainval, 'labels', bboxes_urls[i]), 'r') as f:
            for row in f:
                row = [float(x) for x in row.strip().split(' ')]
                bboxes.append([row[1], row[2], row[3], row[4], row[0]])
        
        trans = run_augments_on_image(os.path.join(dataset_dir, trainval, 'images',  batch_img_names[i]), bboxes) 

        img_index = len(os.listdir(os.path.join(dataset_dir, 'train' , 'images'))) + len(os.listdir(os.path.join(dataset_dir, 'valid', 'images'))) + 1
        
        for j in range(len(trans)):
            img_trans, bboxes_trans = trans[j]   
            Image.fromarray(img_trans).save(os.path.join(dataset_dir, trainval, 'images' , f'image-{img_index}.{batch_img_names[j].split(".")[-1]}')) 

            with open(os.path.join(dataset_dir, trainval, 'labels', f'image-{img_index}.txt'), 'w') as f:
                for boxs in bboxes_trans:
                    print(f'{int(boxs[4])} {boxs[0]} {boxs[1]} {boxs[2]} {boxs[3]}', file=f)

            num_images += 1
            img_index += 1

            if num_images >= max_images_to_generate:
                break
    
        if num_images >= max_images_to_generate:
            break

    print(f'Data augmentation complete. {num_images} images have been saved to {os.path.join(dataset_dir, trainval)}')