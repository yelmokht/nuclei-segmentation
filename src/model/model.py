from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, BatchNormalization, Activation
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.metrics import Precision, Recall # type: ignore
from keras.models import load_model
import os

from model.augmentation import generate_augmented_data, train_val_split
from model.data import load_train_data
from model.pre_processing import preprocess_data_1
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
from segmentation_models.losses import JaccardLoss
from segmentation_models.metrics import IOUScore, FScore, Precision, Recall
import tensorflow as tf
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

def jaccard_loss(y_true, y_pred):
    intersection = tf.reduce_sum(y_true * y_pred, axis=(1,2))
    union = tf.reduce_sum(y_true + y_pred, axis=(1,2)) - intersection
    jaccard = (intersection + 1e-15) / (union + 1e-15)
    return 1 - tf.reduce_mean(jaccard)

def unet_model():
    print('Creating model')
    model = UNet(INPUT_SHAPE).build()
    model.compile(optimizer=Adam(), loss=JaccardLoss(), metrics=['accuracy'])
    print('Model created successfully !')
    # model.summary()
    return model

def modified_unet_model():
    print('Creating model')
    model = sm.Unet()
    model.compile(optimizer=Adam(), loss=JaccardLoss(), metrics=[IOUScore(), 'accuracy', FScore(beta=1), Precision(), Recall()])
    print('Model created successfully !')
    # model.summary()
    return model


def train_model(model, train_generator, val_generator, epochs, steps_per_epoch, validation_steps):
    history = model.fit_generator(train_generator, validation_data=val_generator, steps_per_epoch=steps_per_epoch, validation_steps=validation_steps, epochs=epochs)
    return model, history

def save_model(model, file_path):
    print(f'Saving model to {file_path}')
    model.save(file_path)
    print('Model saved successfully !')

def save_history(history, file_path):
    print(f'Saving history {file_path}')
    history_df = pd.DataFrame(history.history)
    history_df.to_csv(file_path, index=False)
    history = history_df.to_dict(orient='list')
    print('History saved successfully !')
    return history

def load_unet_model(model_name):
    print(f'Loading model {model_name}')
    model_path = './models/' + f'{model_name}/' + 'model.h5'
    model = load_model(model_path)
    print(f'Model {model_name} loaded successfully.')
    return model

def load_history(model_name):
    print(f'Loading history of {model_name}')
    history_path = './models/' + model_name + '/history.csv'
    history_df = pd.read_csv(history_path)
    history = history_df.to_dict(orient='list')
    print(f'History of {model_name} loaded successfully !')
    return history

def train(model_name, batch_size, epochs):
    train_images, train_masks, = load_train_data(TRAIN_PATH)
    pp_train_images, pp_train_masks= preprocess_data_1(train_images, train_masks)
    pp_train_images, pp_val_images, pp_train_masks, pp_val_masks = train_val_split(pp_train_images, pp_train_masks, ratio=0.3)
    train_generator, val_generator, steps_per_epoch, validation_steps = generate_augmented_data(pp_train_images, pp_train_masks, pp_val_images, pp_val_masks, batch_size)
    model = modified_unet_model()
    print(f"Start training model {model_name} \n")
    model, history = train_model(model, train_generator, val_generator, epochs, steps_per_epoch, validation_steps)
    dir_path = f"./models/{model_name}"
    os.makedirs(dir_path)
    model_path = f"./models/{model_name}/model.h5"
    save_model(model, model_path)
    history_path = f"./models/{model_name}/history.csv"
    save_history(history, history_path)
    print(f"Model {model_name} trained successfully !")