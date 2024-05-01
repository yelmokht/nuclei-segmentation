# from keras.models import Model
# from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Activation
# from keras.layers import MaxPool2D
# from keras.metrics import MeanIoU
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.metrics import Precision, Recall
# from tensorflow.keras.metrics import MeanIoU
# from tensorflow import keras
# from keras.optimizers import Adam, SGD
# import os
# os.environ["SM_FRAMEWORK"] = "tf.keras"
# import segmentation_models as sm
# from segmentation_models.losses import bce_jaccard_loss, JaccardLoss, bce_dice_loss
# from segmentation_models.metrics import IOUScore, FScore, Precision, Recall
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import load_model
import pandas as pd
from skimage.io import imread
from glob import glob
from tqdm.auto import tqdm
import numpy as np
from skimage.measure import regionprops, label
from skimage.transform import resize
import matplotlib.pyplot as plt




def read_image(image_path):
    return imread(image_path).astype(np.uint8)

def main():
    # model = load_model("../model/best_model.keras", compile=False)
    img = read_image("../data/image.png")
    img = resize(img, (256, 256), mode='constant', preserve_range=True)
    img = img.astype(np.uint8)
    plt.imshow(img)
    plt.show()
    # prediction = model.predict("../data/image.png")




if __name__ == "__main__":
    main()