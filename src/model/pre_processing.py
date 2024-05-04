from model.config import *
import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray
from tqdm import tqdm

def preprocess(image, shape, param=False):
    processed_image = None

    if shape[2] == 3:
        if len(image.shape) == 3 and image.shape[2] >= shape[2]:
            processed_image = resize(image[:, :, :shape[2]], (shape[0], shape[1])).astype(np.float32)
        else:
            print(image.shape)
            processed_image = np.repeat(image[:, :, np.newaxis], 3, axis=-1)
            print(processed_image.shape)
            processed_image = resize(processed_image[:, :, :shape[2]], (shape[0], shape[1])).astype(np.float32)

    if shape[2] == 1:
        processed_image = resize(image, (shape[0], shape[1])).astype(np.float32)
        if param == False:
            processed_image = np.expand_dims(processed_image, axis=-1)

    return processed_image.astype(np.float32)

def preprocess_data(train_images, test_1_images, test_2_images, train_masks, test_1_masks, test_2_masks):
    train_images = np.array([preprocess(image, IMAGE_SHAPE) for image in tqdm(train_images, desc='Preprocess train images')])
    test_1_images = np.array([preprocess(image, IMAGE_SHAPE) for image in tqdm(test_1_images, desc='Preprocess test images (stage 1)')])
    test_2_images = np.array([preprocess(image, IMAGE_SHAPE) for image in tqdm(test_2_images, desc='Preprocess test images (stage 2)')])
    train_masks = np.array([preprocess(mask, MASK_SHAPE) for mask in tqdm(train_masks , desc='Preprocess train masks')])
    test_1_masks = np.array([preprocess(mask, MASK_SHAPE) for mask in tqdm(test_1_masks, desc='Preprocess test masks (stage 1)')])
    test_2_masks = np.array([preprocess(mask, MASK_SHAPE) for mask in tqdm(test_2_masks, desc='Preprocess test masks (stage 2)')])
    return train_images, test_1_images, test_2_images, train_masks, test_1_masks, test_2_masks
