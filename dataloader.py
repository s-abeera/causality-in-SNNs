import sonnet as snt
import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
import json
import os


class ACREDataset: 
    """
    Loads the ACRE dataset and returns a batch of images and labels. The class is an iterator, so each call to next() returns a new batch
    of images and labels. The class also has a method to load all the labels for the dataset.

    Args:
        batch_size (int): Number of samples in a batch
        dataset (str): Name of the dataset to load. Can be 'train', 'val', or 'test'
        data_path (str): Path to the dataset folder
        img_size (tuple): Size of the images in the dataset. Default is (128, 128)
        normalized (bool): Whether to normalize the images or not. Default is False
    """

    def __init__(self, batch_size, dataset:str, data_path = 'images', img_size = (128, 128), normalized = False):
        self.batch_size = batch_size 
        self.shift = self.batch_size * 10
        self.dataset = dataset
        self.data_path = data_path
        #self.image_path = f'{data_path}\images'
		#self.config_path = f'{data_path}\config'
        self.img_size = img_size
        self.normalized = normalized
        self.current_batch = 0

        data = self.load_data_stack(dataset)
        batched_dataset = data.batch(self.batch_size)
        self.images = iter(batched_dataset)

        self.labels = tf.convert_to_tensor(self.load_labels(dataset))
        self.n_batches = self.labels.shape[0] // self.batch_size
        
    def preprocess_image(self, filename):
        """
        Preprocesses the image by reading it from the file, resizing it, and normalizing it, if not already normalized.

        Args:
            filename (str): Path to the image file

        Returns:
            image (tf.Tensor): Tensor of the image
        """

        image_string = tf.io.read_file(filename)
        image = tf.image.decode_png(image_string, channels=4)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, self.img_size)
        if self.normalized:
             image = tf.image.per_image_standardization(image)
        return image

    
    def load_data_stack(self, dataset: str, window_size = 10): 
        """
        Loads the data from the dataset folder and stacks it into a dataset trials instead of individual images.
        For example, the training set loads 60,000 images, the function stacks them into 6,000 trials of 10 images each.

        Args:
            dataset (str): Name of the dataset to load. Can be 'train', 'val', or 'test'
            window_size (int): Number of images to stack into a trial. Default is 10

        Returns:
            stacked_dataset (tf.data.Dataset): Dataset of trials of images
        """

        file_paths = [os.path.join(self.data_path, file) for file in os.listdir(self.data_path) if file.endswith('.png') and dataset in file]
        #file_paths = [os.path.join(self.image_path, file) for file in os.listdir(self.image_path) if file.endswith('.png') and dataset in file]
        dataset = tf.data.Dataset.from_tensor_slices((file_paths))
        dataset = dataset.map(self.preprocess_image)
        windowed_dataset = dataset.window(window_size, shift=window_size)
        stacked_dataset = windowed_dataset.flat_map(lambda x: x.batch(window_size))

        return stacked_dataset
    
    def load_labels(self, dataset: str):
        """
        Loads the labels for the dataset from the config folder. The labels are stored in a json file.

        Args:
            dataset (str): Name of the dataset to load. Can be 'train', 'val', or 'test'

        Returns:
            labels (list): List of labels for each trial in the dataset. Each trial has 6 context images and 4
            query images, therefore each trial has 4 labels. 
            The labels are in the shape (num of batches, 4)
        """

        file = f'config\{dataset}.json'
        #file = f'{self.config_path}\{dataset}.json'
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
        try:
            batch = self.images.next()
        except StopIteration:
            batch = self.images.next()
   
        labels = self.labels[self.current_batch:self.current_batch + self.batch_size]
        self.current_batch += 1

        return batch, labels
    

if __name__ == '__main__':

	acre = ACREDataset(5, 'train')
	training_batch, label = acre.get_next_batch()
	print(training_batch.shape, label.shape)


	for data in acre:
		training_batch, label = acre.get_next_batch()
		print(training_batch.shape, label.shape)
  
