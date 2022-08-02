import os

class AnnotationDataLoader:
	def __init__(self, labels, dataset_dir):
		self.labels = labels
		self.starting_num_images = len(os.listdir(os.path.join(dataset_dir, 'train' , 'images'))) + len(os.listdir(os.path.join(dataset_dir, 'valid' , 'images'))) 
		self.current_num_images = self.starting_num_images #TODO: update methods to actively track the current_num_images
		self.batch_ptr = 0
		self.input_dir = dataset_dir

		self.img_batches , self.label_batches = self.batch_images(self.labels, starting_img_per_batch = 50)

	def update_number_images_taken(self, num_images_taken, absolute = False):
		if not absolute:
			self.current_num_images += num_images_taken
		else:
			self.current_num_images = num_images_taken

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

	def get_next_batch(self):
		if not self.has_next_batch():
			raise Exception("Trying to access a batch that doesn't exist. The current len of self.img_batches is 0.")

		self.batch_ptr += 1
		return (self.img_batches[self.batch_ptr - 1], self.label_batches[self.batch_ptr - 1])
	
	def reset_batch_ptr(self):
		self.batch_ptr = 0

	def clear_batches(self):
		self.img_batches.clear()
		self.label_batches.clear()
		self.reset_batch_ptr()

	def set_data_and_batch_evenly(self, images, labels ):

		if len(images) != len(labels):
			raise Exception(f'There must be an equal number of images and labels, currently we have {len(images)} images and {len(labels)} labels')

		img_batches = []
		label_batches = []
		
		imgs_per_batch = min(ceil(len(images) / len(self.labels), 250))
		num_batches = ceil(len(images) / imgs_per_batch)
		lst_index = 0

		for j in range(num_batches):
			img_batches.append(images[lst_index : min(lst_index + imgs_per_batch, len(images))])
			label_batches.append( labels[lst_index : min(lst_index + imgs_per_batch, len(images))]                )
			lst_index += imgs_per_batch

			print(f'Batch {len(img_batches)} complete.')
			print(f'Batch size: {len(img_batches[-1])}')
		
		return img_batches , label_batches

	def batch_images(self, classnames, starting_img_per_batch = 50):
		img_batches = []
		label_batches = []

		#Precompute some information to increase efficiency
		completed = [False for i in range(len(classnames))]
		last_image_index = [0 for i in range(len(classnames))]
		total_files_per_class = [len(os.listdir(os.path.abspath(os.path.join(self.input_dir, classname)))) for classname in classnames]
		abs_files = [os.listdir(os.path.abspath(os.path.join(self.input_dir, classname))) for classname in classnames] 
		
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
