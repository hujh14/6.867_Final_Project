from tensorflow.examples.tutorials.mnist import input_data

from matplotlib import pyplot as plt
import numpy as np

class Data:

    def __init__(self, images):
        self.images = images
        self.n, self.image_size, _ = self.images.shape

    def random_masks(self, n):
        masks = np.zeros((n,self.image_size, self.image_size))
        for i in xrange(n):
            masks[i] = self.generate_random_mask()
        return masks

    def generate_random_mask(self):
        max_mask = self.image_size/2
        mask_size = np.random.randint(max_mask/2, max_mask)

        x = np.random.randint(self.image_size-mask_size)
        y = np.random.randint(self.image_size-mask_size)
        mask = np.ones((self.image_size, self.image_size))
        for i in xrange(mask_size):
            for j in xrange(mask_size):
                mask[x+i][y+j] = 0
        return mask

    def open_image(self, image):
        plt.imshow(image, cmap = plt.get_cmap('gray'))
        plt.show()

    def open_image_with_mask(self, image, mask):
        self.open_image(mask)
        self.open_image(np.multiply(image, mask))


if __name__ == "__main__":
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    data = mnist.train.next_batch(50)[0]
    images = data.reshape((50,28,28))
    data = Data(images)

    masks = data.random_masks(2)
    for i in xrange(10):
        data.open_image_with_mask(data.images[i], masks[i])
