from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, BatchNormalization, Activation
from keras.models import Model, load_model
from keras.callbacks import Callback
from keras.optimizers import Adam, SGD
import os

import tensorflow as tf
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
from segmentation_models.losses import JaccardLoss
from segmentation_models.metrics import IOUScore, FScore, Precision, Recall
import pandas as pd
from model.config import *

class UNet:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def conv_block(self, input, num_filters):
        x = Conv2D(num_filters, 3, padding="same")(input)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(num_filters, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        return x

    def encoder_block(self, input, num_filters):
        x = self.conv_block(input, num_filters)
        p = MaxPooling2D((2, 2))(x)
        return x, p

    def decoder_block(self, input, skip_features, num_filters):
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
        x = concatenate([x, skip_features])
        x = self.conv_block(x, num_filters)
        return x

    def build(self):
        inputs = Input(self.input_shape)

        s1, p1 = self.encoder_block(inputs, 64)
        s2, p2 = self.encoder_block(p1, 128)
        s3, p3 = self.encoder_block(p2, 256)
        s4, p4 = self.encoder_block(p3, 512)

        b1 = self.conv_block(p4, 1024)

        d1 = self.decoder_block(b1, s4, 512)
        d2 = self.decoder_block(d1, s3, 256)
        d3 = self.decoder_block(d2, s2, 128)
        d4 = self.decoder_block(d3, s1, 64)

        outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)
        model = Model(inputs, outputs, name="U-Net")

        return model

def unet_model():
    model = UNet(INPUT_SHAPE).build()
    model.compile(optimizer=Adam(), loss=JaccardLoss(), metrics=[IOUScore(), 'accuracy', FScore(beta=1), Precision(), Recall()])
    # model.summary()
    return model

def modified_unet_model():
    model = sm.Unet('resnet152', classes=1, activation='sigmoid', input_shape=INPUT_SHAPE)
    model.compile(optimizer=Adam(), loss=JaccardLoss(), metrics=[IOUScore(), 'accuracy', FScore(beta=1), Precision(), Recall()])
    # model.summary()
    return model

def train_model(model, train_generator, val_generator, epochs, steps_per_epoch, validation_steps):
    history = model.fit_generator(train_generator, validation_data=val_generator, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, epochs=epochs)
    return model, history

def save_model(model, file_path):
    model.save(file_path)
    
def save_history(history, file_path):
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(file_path, index=False)
    history = history_df.to_dict(orient='list')
    return history

def load_unet_model(model_name):
    model_path = MODELS_PATH + f'{model_name}/' + MODEL_FORMAT
    model = load_model(model_path, compile=False)
    return model

def load_history(model_name):
    history_path = MODELS_PATH + f'{model_name}/'+ HISTORY_FORMAT
    history_df = pd.read_csv(history_path)
    history = history_df.to_dict(orient='list')
    return history