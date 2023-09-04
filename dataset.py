import collections
#import environments
import sonnet as snt
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
# import skimage
# from skimage import data
# from skimage import transform
# import os 
import tensorflow_datasets as tfds
import json

import os
tf.autograph.set_verbosity(0)


#img = mpimg.imread('images\ACRE_train_000000_00.png')
#print(type(img))
##checking range of img
#print(img[0][0])
#
#print(np.max(img))
#
#n = 100
#img_size = (128, 128)
#
## Path to the 'images' folder
#folder_path = 'images'
#
## Get a list of all image file paths in the folder
#file_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith('.png') and 'train' in file]
#
## Take the first 50 file paths
## file_paths = file_paths[:n]
#print(len(file_paths))
#
#
## Create a dataset from the file paths
#dataset = tf.data.Dataset.from_tensor_slices(file_paths)
#
#def preprocess_image(filename):
#	 image_string = tf.io.read_file(filename)
#	 image = tf.image.decode_png(image_string, channels=4)
#	 image = tf.image.convert_image_dtype(image, tf.float32)
#	 image = tf.image.resize(image, img_size)
#	 return image
#
#dataset = dataset.map(preprocess_image)
#
## Create a dataset of 10-image windows
#windowed_dataset = dataset.window(10, shift=10)
#
## Flatten the dataset of windows
#stacked_dataset = windowed_dataset.flat_map(lambda x: x.batch(10))
#
#print(stacked_dataset)
#
#
#batched_dataset = stacked_dataset.batch(5)
#
#for batch in batched_dataset.take(2):
#	 print(batch.shape)
#
#def show_img(image):
#	 fig, axes = plt.subplots(1, 10, figsize=(10, 10))
#	 axes = axes.flatten()
#	 for i, image in enumerate(image):
#	 # Plot the image in the corresponding subplot
#		 axes[i].imshow(image)
#		 axes[i].axis('off')
#
#
## img1 = images[0]
## # img2 = image2[0]
## # img3 = image1[2]
## show_img(img1)
## show_img(img2)
## show_img(img3)
#
#a = iter(batched_dataset)
#a.next()



class ACREDataset: 
	def __init__(self, batch_size, dataset:str, data_path = 'images', img_size = (128, 128), normalized = True):
		
		self.batch_size = batch_size 
		self.shift = self.batch_size * 10
		self.dataset = dataset
		self.data_path = data_path
		self.image_path = f'{data_path}\images'
		self.config_path = f'{data_path}\config'
		self.img_size = img_size
		self.normalized = normalized
		self.current_batch = 0

		data = self.load_data_stack(dataset)
		batched_dataset = data.batch(self.batch_size)
		
		self.images = iter(batched_dataset)
		self.labels = tf.convert_to_tensor(self.load_labels(dataset), dtype=tf.int64)
		
		self.n_batches = self.labels.shape[0]
		
	def preprocess_image(self, filename):
		image_string = tf.io.read_file(filename)
		image = tf.image.decode_png(image_string, channels=4)
		image = tf.image.convert_image_dtype(image, tf.float32)
		image = tf.image.resize(image, self.img_size)
		return image

	
	def load_data_stack(self, dataset: str, window_size = 10): 
		#dataset can be train, val, test

		file_paths = [os.path.join(self.image_path, file) for file in os.listdir(self.image_path) if file.endswith('.png') and dataset in file]
		#file_paths = file_paths[self.current_batch[dataset] * self.shift: (self.current_batch[dataset] + 1) * self.shift]

		dataset = tf.data.Dataset.from_tensor_slices((file_paths))
		dataset = dataset.map(self.preprocess_image)
		windowed_dataset = dataset.window(window_size, shift=window_size)
		stacked_dataset = windowed_dataset.flat_map(lambda x: x.batch(window_size))

		return stacked_dataset
	
	def load_labels(self, dataset: str):
		file = f'{self.config_path}\{dataset}.json'
		labels = []
		
		with open(file, 'r') as json_File:
			sample_load_file = json.load(json_File)

		for sample in sample_load_file:
			trial_labels = []
			for i in np.arange(6, 10):
				trial_labels.append(sample[i]['label'])
			labels.append(trial_labels)

		return labels
	
	def get_next_batch(self):
		batch = self.images.next()
		labels = self.labels[self.current_batch:self.current_batch+self.batch_size]
		self.current_batch += 1

		return batch, labels
	
	# def get_next_validation_batch(self):
	#	  batch = self.images.next()
	#	  labels = self.labels[self.current_batch['train']]
	#	  self.current_batch['val'] += 1


	# def get_next_test_batch(self):
	#	  self.current_batch['test'] += 1
	#	  dataset = self.load_data_stack('test')
	#	  batch = self.load_batch_images(dataset)
	#	  return batch, self.current_batch['test'] 
  


if __name__ == '__main__':

	acre = ACREDataset(5, 'train')
	training_batch, label = acre.get_next_batch()
	print(training_batch.shape, label.shape)


	for data in acre:
		training_batch, label = acre.get_next_batch()
		print(training_batch.shape, label.shape)

	
	
