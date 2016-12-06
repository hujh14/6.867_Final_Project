import numpy as np

def crop_center(img, d):
    width, height = img.size

    x = width/2 - d/2
    y = height/2 - d/2
    box = (x,y,x+d,y+d)

    crop = img.copy()
    crop = crop.crop(box)

    cropped_image = img.copy()
    pixels = cropped_image.load()
    for i in xrange(d):
        for j in xrange(d):
            pixels[x+i,y+j] = (0,0,0)
    return cropped_image, crop, box


def crop_random(img):
    width, height = img.size

    d = np.random.randint(10, 20)
    x = np.random.randint(0, width - d)
    y = np.random.randint(0, height - d)
    box = (x,y,x+d,y+d)

    crop = img.copy()
    crop = crop.crop(box)

    cropped_image = img.copy()
    pixels = cropped_image.load()
    for i in xrange(d):
        for j in xrange(d):
            pixels[x+i,y+j] = (0,0,0)
    return cropped_image, crop, box