import numpy as np
from matplotlib.image import imsave
import errno
import os


def read_labels(path_to_labels):
    with open(path_to_labels, 'rb') as f:
        labels = np.fromfile(f, dtype=np.uint8)
        return labels


def read_all_images(path_to_images):
    with open(path_to_images, 'rb') as f:
        everything = np.fromfile(f, dtype=np.uint8)
        images = np.reshape(everything, (-1, 3, 96, 96))
        images = np.transpose(images, (0, 3, 2, 1))
        return images


def save_image(image, name):
    imsave('%s.png' % name, image, format='png')


def save_images(images, labels):
    i = 0
    for image in images:
        label = labels[i]
        directory = './input_1/stl/test/' + str(label) + '/'
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                pass
        filename = directory + str(i)
        save_image(image, filename)
        i = i + 1
