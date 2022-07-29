import os
from argparse import Namespace
from yahoo_images import yahoo
from shutterstock_images import shutterstock
from google_images import google
import numpy as np
import cv2
from PIL import Image


class WebDataLoader:
	def __init__(self, MAX_IMAGES, MAX_TRAIN_IMAGES, labels, dataset_dir, PHOTOS_DIR = 'photos'):
		self.MAX_IMAGES = MAX_IMAGES
		self.labels = labels
		self.MAX_TRAIN_IMAGES = MAX_TRAIN_IMAGES
		self.PHOTOS_DIR = PHOTOS_DIR
		self.OUTPUT_DIR = PHOTOS_DIR
		self.starting_num_images = len(os.listdir(os.path.join(dataset_dir, 'train' , 'images'))) + len(os.listdir(os.path.join(dataset_dir, 'valid' , 'images')))
		self.current_num_images = self.starting_num_images
		self.batch_ptr = 0

		# For the initial version, we will only scrape once, and get as many images as possible. The code has been modularized to scale, though.
		# uncomment below line in final vers, commented for debugging ease
		self.download_by_chunk(self.labels, self.MAX_IMAGES, ignore_excess = False)
		self.img_batches , self.label_batches = self.batch_images(self.labels, starting_img_per_batch = 50)

	
	def download_images_from_shutterstock(self, classname, num_images):
		label_out_dir = os.path.abspath(os.path.join('scraper', self.OUTPUT_DIR, classname))
		print(f'Shutterstock scraper is downloading images to {label_out_dir}')

		if not os.path.exists(label_out_dir):
			os.makedirs(label_out_dir)

		shutterstock.download_images(classname,
							num_images,
							output_dir=label_out_dir,
							pool_size=10,
							file_type="",
							force_replace=False,
							extra_query_params='')

	
	def download_images_from_yahoo(self, classname, num_images):
		label_out_dir = os.path.abspath(os.path.join('scraper', self.OUTPUT_DIR, classname))
		print(f'Yahoo scraper is downloading images to {label_out_dir}')

		if not os.path.exists(label_out_dir):
			os.makedirs(label_out_dir)

		yahoo.download_images(classname,
							num_images,
							output_dir=label_out_dir,
							pool_size=10,
							file_type="",
							force_replace=False,
							extra_query_params='')
			

	def download_images_from_google(self, classname, num_images):
		label_out_dir = os.path.abspath(os.path.join('scraper', self.OUTPUT_DIR, classname))
		print(f'Google scraper is downloading images to {label_out_dir}')

		if not os.path.exists(label_out_dir):
			os.makedirs(label_out_dir)

		google.download_images(classname,
							num_images,
							output_dir=label_out_dir,
							pool_size=10,
							file_type="",
							force_replace=False,
							extra_query_params='')


	def download_by_chunk(self, classnames, MAX_IMAGES, ignore_excess = False):

		images_per_label = MAX_IMAGES // len(classnames)
		print(f'Downloading {images_per_label} images for each category.')

		# shutterstock, yahoo, google 
		num_scrapers = 3 

		cur_image_count = [0] * len(classnames)
		images_per_scraper = images_per_label // num_scrapers + 1

		# Yahoo
		for i in range(len(classnames)):
			if cur_image_count[i] < images_per_label:
				self.download_images_from_yahoo(classnames[i], images_per_scraper)
				cur_image_count[i] = len(os.listdir(os.path.abspath(os.path.join('scraper', self.OUTPUT_DIR, classnames[i]))))
		
		# Google
		for i in range(len(classnames)):
			if cur_image_count[i] < images_per_label:
				self.download_images_from_google(classnames[i], images_per_scraper)
				cur_image_count[i] = len(os.listdir(os.path.abspath(os.path.join('scraper', self.OUTPUT_DIR, classnames[i]))))

		# Shutterstock
		for i in range(len(classnames)):
			if cur_image_count[i] < images_per_label:
				self.download_images_from_shutterstock(classnames[i], images_per_label - cur_image_count[i])
				cur_image_count[i] = len(os.listdir(os.path.abspath(os.path.join('scraper', self.OUTPUT_DIR, classnames[i]))))

		if not ignore_excess:
			print('Excess has been specified to be removed, set ignore_excess to be True if the extra images is wanted')
			[os.remove(os.path.abspath(os.path.join('scraper', self.OUTPUT_DIR, classnames[i], f))) for f in os.listdir(os.path.abspath(os.path.join('scraper', self.OUTPUT_DIR, classnames[i])))[images_per_label:] ]

	def batch_images(self, classnames, starting_img_per_batch = 50):
		img_batches = []
		label_batches = []

		#Precompute some information to increase efficiency
		completed = [False for i in range(len(classnames))]
		last_image_index = [0 for i in range(len(classnames))]
		total_files_per_class = [len(os.listdir(os.path.abspath(os.path.join('scraper' , self.PHOTOS_DIR, classname)))) for classname in classnames]
		abs_files = [os.listdir(os.path.abspath(os.path.join('scraper', self.PHOTOS_DIR, classname))) for classname in classnames] 
		
		print('Total files: ', sum(total_files_per_class))

		#While we haven't exhausted all images
		while sum(completed) < len(classnames):

			image_batch = []
			label_batch = []                

			for i in range(len(classnames)):            
				if not completed[i]:
					image_batch.extend( [os.path.abspath(os.path.join('scraper', self.OUTPUT_DIR, classnames[i], fname)) for fname in abs_files[i][last_image_index[i] : min(last_image_index[i] + starting_img_per_batch, total_files_per_class[i])]])
					label_batch.extend([ classnames[i] for j in range(last_image_index[i] , min(last_image_index[i] + starting_img_per_batch, total_files_per_class[i]) ) ])
					if last_image_index[i] + starting_img_per_batch >= total_files_per_class[i]:
						completed[i] = True

					last_image_index[i] = min(last_image_index[i] + starting_img_per_batch, total_files_per_class[i])

					

			img_batches.append(image_batch)
			label_batches.append(label_batch)
			starting_img_per_batch = min(starting_img_per_batch * 2, 256 // len(classnames))#


			print(f'Batch {len(img_batches)} complete.')
			print(f'Batch size: {len(img_batches[-1])}')


		return (img_batches, label_batches)

	def get_total_ds_imgs(self):
		return self.current_num_images
	
	def get_total_ds_imgs_added(self):
		return self.get_total_ds_imgs() - self.starting_num_images

	def has_next_batch(self):
		return len(self.img_batches) > 0 and self.batch_ptr < len(self.img_batches)
	
	def get_next_batch_type(self):
		if not self.has_next_batch():
			raise Exception("Trying to access a batch that doesn't exist. The current len of self.img_batches is 0.")

		return "valtrain" if self.get_total_ds_imgs() < self.MAX_TRAIN_IMAGES else "val"

	def update_number_images_taken(self, num_images_taken):
		self.current_num_images += num_images_taken

	def get_next_batch(self):
		if not self.has_next_batch():
			raise Exception("Trying to access a batch that doesn't exist. The current len of self.img_batches is 0.")

		self.batch_ptr += 1
		return (self.img_batches[self.batch_ptr - 1], self.label_batches[self.batch_ptr - 1])
	
	def reset_batch(self):
		self.batch_ptr = 0

class AnnotationDataLoader:
	def __init__(self, labels, dataset_dir):
		self.labels = labels
		self.starting_num_images = len(os.listdir(os.path.join(dataset_dir, 'train' , 'images'))) + len(os.listdir(os.path.join(dataset_dir, 'valid' , 'images')))
		self.current_num_images = self.starting_num_images
		self.batch_ptr = 0

		#For the initial version, we will only scrape once, and get as many images as possible. The code has been modularized to scale, though.
		self.download_by_chunk(self.labels, self.MAX_IMAGES, ignore_excess = False)
		self.img_batches , self.label_batches = self.batch_images(self.labels, starting_img_per_batch = 50)
