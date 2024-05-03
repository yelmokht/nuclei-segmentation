from sklearn.model_selection import train_test_split
import random
from keras.preprocessing.image import ImageDataGenerator

def train_val_split(train_images, train_masks, ratio):
    return train_test_split(train_images, train_masks, test_size=ratio, random_state=0)

def generate_augmented_data(train_images, train_masks, val_images, val_masks):
    img_data_gen_args = dict(rotation_range=90,
                        width_shift_range=0.3,
                        height_shift_range=0.3,
                        shear_range=0.3,
                        zoom_range=0.3,
                        horizontal_flip=True,
                        vertical_flip=True,
                        fill_mode='reflect'
                        )

    mask_data_gen_args = dict(rotation_range=90,
                        width_shift_range=0.3,
                        height_shift_range=0.3,
                        shear_range=0.3,
                        zoom_range=0.3,
                        horizontal_flip=True,
                        vertical_flip=True,
                        fill_mode='reflect',
                        preprocessing_function = lambda x: np.where(x>0, 1, 0).astype(x.dtype))

    # Data augmentation
    image_data_generator = ImageDataGenerator(**img_data_gen_args)
    mask_data_generator = ImageDataGenerator(**mask_data_gen_args)

    # Data augmentation
    image_data_generator.fit(train_images, augment=True, seed=SEED)
    mask_data_generator.fit(train_masks, augment=True, seed=SEED)

    # Train
    train_image_generator = image_data_generator.flow(train_images, batch_size=BATCH_SIZE, shuffle=True, seed=SEED)
    train_mask_generator = mask_data_generator.flow(train_masks, batch_size=BATCH_SIZE, shuffle=True, seed=SEED)

    # Val
    val_image_generator = image_data_generator.flow(val_images, batch_size=BATCH_SIZE, shuffle=True, seed=SEED)
    val_mask_generator = mask_data_generator.flow(val_masks, batch_size=BATCH_SIZE, shuffle=True, seed=SEED)

    # Generator
    train_generator = zip(train_image_generator, train_mask_generator)
    val_generator = zip(val_image_generator, val_mask_generator)

    steps_per_epoch = 2*len(train_image_generator)
    validation_steps = 2*len(val_image_generator)

    print('Batch size: ', BATCH_SIZE)
    print('Steps per epoch: ', steps_per_epoch)
    print('Validation steps: ', validation_steps)

    return train_generator, val_generator, steps_per_epoch, validation_steps