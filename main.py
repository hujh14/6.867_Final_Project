from PIL import Image
from utils import *

import os

image_dir = "./images/imagenet_128"
output_dir_path = "./images/temp"
for filename in os.listdir(image_dir):
    image_path = os.path.join(image_dir, filename)

    image_number = os.path.basename(image_path).split("_")[0]
    output_image_name = "{}_c.png".format(image_number)
    output_image_name2 = "{}_c-.png".format(image_number)
    output_path = os.path.join(output_dir_path, output_image_name)
    output_path2 = os.path.join(output_dir_path, output_image_name2)


    img = Image.open(image_path)
    # cropped_image, crop, box = crop_center(img, 20)
    cropped_image, crop, box = crop_random(img)
    cropped_image.save(output_path, optimize=True)
    crop.save(output_path2, optimize=True)


