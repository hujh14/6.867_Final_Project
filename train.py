from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

from mnist_data import MnistData
from mask import Mask
from model import Model


num_of_images = 20
nums = [0,1,2,3,4,5,6,7,8,9]
# nums = [7]
data = MnistData(num_of_images, nums)

model = Model()
mask_gen = Mask(28)

iterations = 500
model_name = "all,(16,16,16),images={},i={},masked_loss".format(num_of_images,iterations)
# model_name = "single_number_recovery"


# model_path = model.train(iterations, data, mask_gen, model_name)
model_path = "models/{}.ckpt".format(model_name)

print model_path
model.test(model_path, data, mask_gen)

