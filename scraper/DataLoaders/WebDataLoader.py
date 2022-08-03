import os
from scraper.yahoo_images import yahoo
from scraper.shutterstock_images import shutterstock
from scraper.google_images import google
from scraper.remove_dupes import remove


class WebDataLoader:
	def __init__(self, MAX_IMAGES, IMAGES_PER_LABEL, MAX_TRAIN_IMAGES, labels, queries, dataset_dir, PHOTOS_DIR = 'photos'):
		self.MAX_IMAGES = MAX_IMAGES
		self.labels = labels
		self.IMAGES_PER_LABEL = IMAGES_PER_LABEL
		self.MAX_TRAIN_IMAGES = MAX_TRAIN_IMAGES
		self.PHOTOS_DIR = PHOTOS_DIR
		self.OUTPUT_DIR = PHOTOS_DIR
		self.starting_num_images = len(os.listdir(os.path.join(dataset_dir, 'train' , 'images'))) + len(os.listdir(os.path.join(dataset_dir, 'valid' , 'images')))
		self.current_num_images = self.starting_num_images
		self.batch_ptr = 0

		# For the initial version, we will only scrape once, and get as many images as possible. The code has been modularized to scale, though.
		# uncomment below line in final vers, commented for debugging ease
		self.download_by_chunk(self.labels, queries, ignore_excess = True)
		self.img_batches , self.label_batches = self.batch_images(self.labels, starting_img_per_batch = 50)

	
	def download_images_from_shutterstock(self, classname, query, num_images):
		label_out_dir = os.path.abspath(os.path.join('scraper', self.OUTPUT_DIR, classname))
		print(f'Shutterstock scraper is downloading images to {label_out_dir}')

		if not os.path.exists(label_out_dir):
			os.makedirs(label_out_dir)

		shutterstock.download_images(query,
									num_images,
									output_dir=label_out_dir,
									pool_size=10,
									file_type="",
									force_replace=False,
									extra_query_params='')

	
	def download_images_from_yahoo(self, classname, query, num_images):
		label_out_dir = os.path.abspath(os.path.join('scraper', self.OUTPUT_DIR, classname))
		print(f'Yahoo scraper is downloading images to {label_out_dir}')

		if not os.path.exists(label_out_dir):
			os.makedirs(label_out_dir)

		yahoo.download_images(query,
							num_images,
							output_dir=label_out_dir,
							pool_size=10,
							file_type="",
							force_replace=False,
							extra_query_params='')
			

	def download_images_from_google(self, classname, query, num_images):
		label_out_dir = os.path.abspath(os.path.join('scraper', self.OUTPUT_DIR, classname))
		print(f'Google scraper is downloading images to {label_out_dir}')

		if not os.path.exists(label_out_dir):
			os.makedirs(label_out_dir)

		google.download_images(query,
							num_images,
							output_dir=label_out_dir,
							pool_size=10,
							file_type="",
							force_replace=False,
							extra_query_params='')


	def download_by_chunk(self, classnames, queries, excess_factor = 2 , ignore_excess = False):

		print(f'Downloading {self.IMAGES_PER_LABEL} images for each category.')

		# shutterstock, yahoo, google 
		num_scrapers = 3 

		cur_image_count = [0] * len(classnames)

		#Overshoot images on purpose so that we don't have a shortage
		images_per_scraper = (self.IMAGES_PER_LABEL * excess_factor) // num_scrapers + 1

		# Yahoo
		for i in range(len(classnames)):
			for query in queries[classnames[i]]:
				if cur_image_count[i] < (self.IMAGES_PER_LABEL * excess_factor):
					print(f'Downloading images for query: {query}')
					self.download_images_from_yahoo(classnames[i], query, images_per_scraper)
					cur_image_count[i] = len(os.listdir(os.path.abspath(os.path.join('scraper', self.OUTPUT_DIR, classnames[i]))))
				else:
					break
		
		# Google
		for i in range(len(classnames)):
			for query in queries[classnames[i]]:
				if cur_image_count[i] < (self.IMAGES_PER_LABEL * excess_factor):
					print(f'Downloading images for query: {query}')
					self.download_images_from_google(classnames[i], query, images_per_scraper)
					cur_image_count[i] = len(os.listdir(os.path.abspath(os.path.join('scraper', self.OUTPUT_DIR, classnames[i]))))
				else:
					break

		# Shutterstock
		for i in range(len(classnames)):
			for query in queries[classnames[i]]:
				if cur_image_count[i] < (self.IMAGES_PER_LABEL * excess_factor):
					print(f'Downloading images for query: {query}') 
					self.download_images_from_shutterstock(classnames[i], query, images_per_scraper)
					cur_image_count[i] = len(os.listdir(os.path.abspath(os.path.join('scraper', self.OUTPUT_DIR, classnames[i]))))
				else:
					break

		if not ignore_excess:
			print('Excess has been specified to be removed, set ignore_excess to be True if the extra images is wanted')
			[os.remove(os.path.abspath(os.path.join('scraper', self.OUTPUT_DIR, classnames[i], f))) for f in os.listdir(os.path.abspath(os.path.join('scraper', self.OUTPUT_DIR, classnames[i])))[self.IMAGES_PER_LABEL:] ]

		print('Removing duplicate images ...')
		for classname in classnames:
			dupes = remove(os.path.abspath(os.path.join('scraper', f'photos/{classname}')))
			print(f'{dupes} image(s) have been removed from photos/{classname}.')

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
		return len(self.img_batches) > 0 and self.batch_ptr < len(self.img_batches) and self.get_total_ds_imgs() < self.MAX_IMAGES
	
	def get_next_batch_type(self):
		if not self.has_next_batch():
			raise Exception("Trying to access a batch that doesn't exist. The current len of self.img_batches is 0.")

		return "valtrain" if self.get_total_ds_imgs() < self.MAX_TRAIN_IMAGES else "val"

	def update_number_images_taken(self, num_images_taken, absolute = False):
		if not absolute:
			self.current_num_images += num_images_taken
		else:
			self.current_num_images = num_images_taken

	def get_next_batch(self):
		if not self.has_next_batch():
			raise Exception("Trying to access a batch that doesn't exist. The current len of self.img_batches is 0.")

		self.batch_ptr += 1


		if self.get_total_ds_imgs() + len(self.img_batches[self.batch_ptr - 1]) > self.MAX_IMAGES:
			imgs = self.img_batches[self.batch_ptr - 1][:(self.MAX_IMAGES - self.get_total_ds_imgs())]
			lbls = self.label_batches[self.batch_ptr - 1][:(self.MAX_IMAGES - self.get_total_ds_imgs())]

			self.clear_batches()
			return (imgs, lbls)
		
		elif self.get_total_ds_imgs() + len(self.img_batches[self.batch_ptr - 1]) > self.MAX_TRAIN_IMAGES and self.get_next_batch_type() == "valtrain":
			imgs = self.img_batches[self.batch_ptr - 1][:(self.MAX_TRAIN_IMAGES - self.get_total_ds_imgs())]
			lbls = self.label_batches[self.batch_ptr - 1][:(self.MAX_TRAIN_IMAGES - self.get_total_ds_imgs())]
			
			return (imgs, lbls)

		return (self.img_batches[self.batch_ptr - 1], self.label_batches[self.batch_ptr - 1])
	
	def reset_batch_ptr(self):
		self.batch_ptr = 0

	def clear_batches(self):
		self.img_batches.clear()
		self.label_batches.clear()
		self.reset_batch_ptr()


