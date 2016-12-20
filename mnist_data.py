from tensorflow.examples.tutorials.mnist import input_data

from matplotlib import pyplot as plt
import numpy as np

class MnistData:

    def __init__(self, num_of_images, nums):
        self.mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        images = np.zeros((0,28,28))
        for num in nums:
            num_images = self.getNumber(num_of_images, num)
            images = np.concatenate((images, num_images))
        np.random.shuffle(images)

        self.total_images, self.image_size, _ = images.shape
        if self.total_images == 1:
            self.train_images = images
            self.test_images = images
        else:
            num_of_training_images = int(0.9*self.total_images)
            self.train_images = images[:num_of_training_images]
            self.test_images = images[num_of_training_images:]

    def one_of_each(self):
        images = np.zeros((10,28,28))
        for i in xrange(10):
            num_image = self.getNumber(1,i)
            images[i] = num_image
        return images


    def getNumber(self, num_of_images, num):
        found_images = 0
        images = np.zeros((num_of_images,28,28))
        while True:
            batch = self.mnist.train.next_batch(np.random.randint(10,100))
            x = batch[0]
            y = batch[1]
            for i in xrange(x.shape[0]):
                if np.argmax(y[i]) == num:
                    images[found_images] = x[i].reshape(1,28,28)
                    found_images += 1
                    if found_images == num_of_images:
                        return images




