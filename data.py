from tensorflow.examples.tutorials.mnist import input_data

from matplotlib import pyplot as plt
import numpy as np

class Data:

    def __init__(self, images):
        self.total_images, self.image_size, _ = images.shape
        if self.total_images == 1:
            self.train_images = images
            self.test_images = images
        else:
            num_of_training_images = int(0.9*self.total_images)
            self.train_images = images[:num_of_training_images]
            self.test_images = images[num_of_training_images:]

        # print self.train_images.shape
        # print self.test_images.shape
        # print self.total_images

