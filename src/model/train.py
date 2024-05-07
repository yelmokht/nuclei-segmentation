from model.data import load_train_data
from model.pre_processing import preprocess_data
from model.augmentation import train_val_split
from model.augmentation import generate_augmented_data
from model.config import  HISTORY_FORMAT, TRAIN_PATH, MODELS_PATH, MODEL_FORMAT
from model.model import unet_model, train_model, save_model, save_history
import os

def train(model_name, batch_size, epochs):
    train_images, train_masks, = load_train_data(TRAIN_PATH)
    pp_train_images, pp_train_masks= preprocess_data(train_images, train_masks)
    pp_train_images, pp_val_images, pp_train_masks, pp_val_masks = train_val_split(pp_train_images, pp_train_masks, ratio=0.3)
    train_generator, val_generator, steps_per_epoch, validation_steps = generate_augmented_data(pp_train_images, pp_train_masks, pp_val_images, pp_val_masks, batch_size)
    print(f'Creating model {model_name} ...')
    model = unet_model()
    print(f'Model {model_name} created successfully !')
    print(f"Start training model {model_name} ...\n")
    model, history = train_model(model, train_generator, val_generator, epochs, steps_per_epoch, validation_steps)
    print(f"Model {model_name} trained successfully !")
    dir_path = MODELS_PATH + f'{model_name}'
    os.makedirs(dir_path)
    model_path = MODELS_PATH + f'{model_name}/' + MODEL_FORMAT
    print(f'Saving model to {model_path} ...')
    save_model(model, model_path)
    print(f'Model {model_name} saved successfully !')
    history_path = MODELS_PATH + f'{model_name}/' + HISTORY_FORMAT
    print(f'Saving history to {history_path} ...')
    save_history(history, history_path)
    print('History saved successfully !')