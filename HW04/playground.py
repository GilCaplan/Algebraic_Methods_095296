import pickle
import numpy as np
from PIL import Image


def unpickle(file):
    with open(file, 'rb') as fo:
        images = pickle.load(fo, encoding='bytes')
    return images


def process(s):
    img = Image.fromarray(s.reshape((3, 32, 32)).transpose((1, 2, 0)).astype('uint8')).convert("L")
    return np.array(img)


def process_batch(path):
    pics = unpickle(path)
    # Step 1: convert all images to gray scale
    images = pics[b'data']
    labels = pics[b'fine_labels'] if b'fine_labels' in pics else 0
    reshaped_images = images.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    grayscale_images = []

    for image in reshaped_images:
        img = Image.fromarray(image.astype('uint8')).convert("L")
        grayscale_images.append(np.array(img).reshape(-1))

    return np.array(grayscale_images).reshape(-1, 1024), labels


grey_images_train, train_labels = process_batch(path:= r'.\cifar-100-python\train')
grey_images_test = process_batch(r'.\cifar-100-python\test')

print()
#single_img = Image.fromarray(grayscale_images[1])
#single_img.show()


#single_img_reshaped = np.transpose(np.reshape(single_img, (3, 32, 32)), (1, 2, 0))
# image = Image.fromarray(single_img_reshaped.astype('uint8'))
# image.show()
