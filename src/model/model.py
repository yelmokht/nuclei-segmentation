from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Activation
from keras.layers import MaxPool2D
from keras.metrics import MeanIoU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.metrics import MeanIoU
from tensorflow import keras
from keras.models import load_model
from keras.optimizers import Adam, SGD
import os

from model.augmentation import generate_augmented_data, train_val_split
from model.data import load_data_1
from model.pre_processing import preprocess_data_1
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
from segmentation_models.losses import bce_jaccard_loss, JaccardLoss, bce_dice_loss
from segmentation_models.metrics import IOUScore, FScore, Precision, Recall
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
import pandas as pd

from model.config import *

# Convolution block
def conv_block(input, num_filters):
  x = Conv2D(num_filters, 3, padding="same")(input)
  x = BatchNormalization()(x)
  x = Activation("relu")(x)

  x = Conv2D(num_filters, 3, padding="same")(x)
  x = BatchNormalization()(x)
  x = Activation("relu")(x)

  return x

# Encoder block = convolution block + max pooling
def encoder_block(input, num_filters):
  x = conv_block(input, num_filters)
  p = MaxPooling2D((2, 2))(x)
  return x, p

# Decoder block = upsampling + concatenation + convolution block
def decoder_block(input, skip_features, num_filters):
  x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
  x = concatenate([x, skip_features])
  x = conv_block(x, num_filters)
  return x

def unet(input_shape):
  inputs = Input(input_shape)

  s1, p1 = encoder_block(inputs, 64)
  s2, p2 = encoder_block(p1, 128)
  s3, p3 = encoder_block(p2, 256)
  s4, p4 = encoder_block(p3, 512)

  b1 = conv_block(p4, 1024)

  d1 = decoder_block(b1, s4, 512)
  d2 = decoder_block(d1, s3, 256)
  d3 = decoder_block(d2, s2, 128)
  d4 = decoder_block(d3, s1, 64)

  outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

  model = Model(inputs, outputs, name="U-Net")
  return model

def jaccard_loss(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
    union = tf.reduce_sum(y_true + y_pred, axis=(1,2)) - intersection
    jaccard = (intersection + 1e-15) / (union + 1e-15)
    return 1 - tf.reduce_mean(jaccard)

def unet_model():
    model = unet(INPUT_SHAPE)
    model.compile(optimizer=Adam(), loss=JaccardLoss(), metrics=['accuracy'])
    # model.summary()
    return model

def modified_unet_model():
    print('Creating model')
    model = sm.Unet()
    model.compile(optimizer=Adam(), loss=JaccardLoss(), metrics=[IOUScore(), 'accuracy', FScore(beta=1), Precision(), Recall()])
    # model.summary()
    return model


def train_model(model, train_generator, val_generator, epochs, steps_per_epoch, validation_steps):
    print('Training model')
    history = model.fit_generator(train_generator, validation_data=val_generator, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, epochs=epochs)
    return model, history

def save_model(model, file_path):
    print('Saving model')
    model.save(file_path)

def save_history(history, file_path):
    print('Saving history')
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(file_path, index=False)
    history = history_df.to_dict(orient='list')
    return history

def load_history(model_name):
    history_path = './models/' + model_name + '/history.csv'
    history_df = pd.read_csv(history_path)
    history = history_df.to_dict(orient='list')
    return history

def load_unet_model(model_name):
    model_path = './models/' + f'{model_name}/' + 'model.h5'
    model = load_model(model_path)
    return model

def train(model_name, batch_size, epochs):
    train_images, train_masks, = load_data_1(TRAIN_PATH)
    pp_train_images, pp_train_masks= preprocess_data_1(train_images, train_masks)
    pp_train_images, pp_val_images, pp_train_masks, pp_val_masks = train_val_split(pp_train_images, pp_train_masks, ratio=0.3)
    train_generator, val_generator, steps_per_epoch, validation_steps = generate_augmented_data(pp_train_images, pp_train_masks, pp_val_images, pp_val_masks)
    model = modified_unet_model()
    model, history = train_model(model, train_generator, val_generator, epochs, steps_per_epoch, validation_steps)
    dir_path = f"./models/{model_name}"
    os.makedirs(dir_path)
    model_path = f"./models/{model_name}/model.keras"
    save_model(model, model_path)
    history_path = f"./models/{model_name}/history.csv"
    save_history(history, history_path)
    print(f"Model {model_name} trained successfully.")