from matplotlib import pyplot as plt
import numpy as np

class Mask:

    def __init__(self, image_size):
        self.image_size = image_size

    def create_mask(self, x0,y0,x1,y1):
        x1 = min(self.image_size, x1)
        y1 = min(self.image_size, y1)

        mask = np.ones((self.image_size, self.image_size))
        for i in xrange(x0,x1):
            for j in xrange(y0, y1):
                mask[i][j] = 0
        return mask

    def center_masks(self, n, mask_size):
        masks = np.zeros((n,self.image_size, self.image_size))
        for i in xrange(n):
            masks[i] = self.center_mask(mask_size)
        return masks

    def center_mask(self, mask_size):
        x0 = (self.image_size-mask_size)/2
        y0 = (self.image_size-mask_size)/2
        x1 = x0 + mask_size
        y1 = y0 + mask_size
        return self.create_mask(x0,y0,x1,y1)

    def random_masks(self, n):
        masks = np.zeros((n,self.image_size, self.image_size))
        for i in xrange(n):
            masks[i] = self.random_mask()
        return masks

    def random_mask(self):
        max_mask = self.image_size/2
        mask_size = np.random.randint(max_mask/2, max_mask)

        x0 = np.random.randint(self.image_size-mask_size)
        y0 = np.random.randint(self.image_size-mask_size)
        x1 = x0 + mask_size
        y1 = y0 + mask_size
        return self.create_mask(x0,y0,x1,y1)

    def open_images(self, inputs, outputs):
        show = np.concatenate((inputs[0], outputs[0]), axis=1)
        for i in xrange(1, len(inputs)):
            pair = np.concatenate((inputs[i], outputs[i]), axis=1)
            show = np.concatenate((show, pair), axis=0)
        plt.imshow(show, cmap = plt.get_cmap('gray'))
        plt.show()

    def open_image_with_mask(self, image, mask):
        self.open_image(np.multiply(image, mask))

if __name__ == "__main__":
    mask_gen = Mask(28)
    masks = mask_gen.center_masks(20)
    for i in xrange(3):
        mask_gen.open_image(masks[i])

    # mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    # num_of_images = 1
    # data = mnist.train.next_batch(num_of_images)[0]
    # images = data.reshape((num_of_images,28,28))
    # data = Data(images)

    # masks = data.random_masks(1)
    # for i in xrange(1):
    #     data.open_image_with_mask(images[i], masks[i])

