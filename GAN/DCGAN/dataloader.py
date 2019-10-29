import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display

#loading mniist digit dataset from tensorflow datasets 
(train_images, train_labels), (test_images,test_labels) = tf.keras.datasets.mnist.load_data()

#concatenating training and test datasets as we don't need test dataset explicitly
train_labels=np.hstack((train_labels,test_labels))
train_images=np.vstack((train_images,test_images))
train_images.shape

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5 # Normalize the images to [-1, 1]

BUFFER_SIZE = 70000
BATCH_SIZE = 256

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
