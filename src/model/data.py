import os
import shutil
from zipfile import ZipFile
import numpy as np
import pandas as pd
import os
from skimage.io import imsave
from tqdm import tqdm
import warnings
from skimage.io import imread
from glob import glob
from tqdm import tqdm
import numpy as np
from skimage.measure import regionprops, label
from skimage.transform import resize
from model.config import *

warnings.filterwarnings('ignore', category=UserWarning)

def unzip_and_structure_data(source_path, unzipped_path, destination_path):
    # Unzip files from source to unzipped
    if not os.path.exists(unzipped_path):
        with ZipFile(source_path, 'r') as zip_ref:
            zip_ref.extractall(unzipped_path)
        print("Files unzipped successfully.")
    else:
        print("Unzipped files already exist.")

    # Structure data from unzipped to destination
    if not os.path.exists(destination_path):
        for root, dirs, files in os.walk(unzipped_path):
            for filename in files:
                if filename.endswith(".zip"):
                    zip_file_path = os.path.join(root, filename)
                    output_folder = os.path.join(destination_path, os.path.splitext(filename)[0])
                    with ZipFile(zip_file_path, 'r') as zip_ref:
                        zip_ref.extractall(output_folder)
                    print(f"Unzipped {filename} to {output_folder}")
        print("Data files structured successfully.")
    else:
        print("Data files already exist.")

def rle_decode(rle_list, mask_shape, mask_dtype):
    masks = []
    for j, rle in enumerate(rle_list):
        mask = np.zeros(np.prod(mask_shape), dtype=mask_dtype)
        s = rle.split()
        starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
        starts -= 1
        ends = starts + lengths
        for lo, hi in zip(starts, ends):
            mask[lo:hi] = 255
        masks.append(mask.reshape(mask_shape[::-1]).T)
    return np.array(masks)

def rle_to_masks(csv_path):
    gt_masks = {}
    df = pd.read_csv(csv_path, sep=',')
    image_ids = sorted(list(df['ImageId'].unique()))

    for image_id in image_ids:
        mask_rles = df.loc[df['ImageId'] == image_id]
        rle_list = mask_rles['EncodedPixels']
        heigth, width = pd.unique(mask_rles['Height'])[0], pd.unique(mask_rles['Width'])[0]
        masks = rle_decode(rle_list=rle_list, mask_shape=(heigth, width), mask_dtype=np.uint16)
        gt_masks[image_id] = masks

    return gt_masks

def save_masks(source_path, destination_path, stage):
    # Load masks
    gt_masks = rle_to_masks(source_path)
    already_saved_masks = False

    # Iterate over each key in gt_masks
    for folder_id, masks in tqdm(gt_masks.items(), desc=f'{stage} images'):
        folder_path = os.path.join(destination_path, str(folder_id))
        if not os.path.exists(folder_path):
            raise FileNotFoundError(f"Destination folder '{folder_path}' does not exist.")

        # Create masks folder inside the folder
        masks_folder_path = os.path.join(folder_path, 'masks')
        if not os.path.exists(masks_folder_path):
            os.makedirs(masks_folder_path)  # Create masks folder if it doesn't exist
            # Iterate over each value in gt_masks[key] (list of masks)
            for idx, mask in enumerate(masks):
                # Save mask as an image file in the masks folder
                mask_filename = f'{folder_id}_{idx}.png'
                mask_file_path = os.path.join(masks_folder_path, mask_filename)
                imsave(mask_file_path, mask)
        else:
            already_saved_masks = True
            break

    if already_saved_masks:
        print(f'{stage} masks already saved !')
    else:
        print(f'{stage} masks successfully saved !')

def remove_ignored_images_masks(csv_path, destination_path):
    df = pd.read_csv(csv_path)
    ignore_images = df.loc[df['Usage'] == 'Ignored', 'ImageId'].tolist()

    for image_id in ignore_images:
        folder_path = os.path.join(destination_path, image_id)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
        else:
            print(f"Folder '{folder_path}' doesn't exist.")
        df = df[df['ImageId'] != image_id]

    df.to_csv(csv_path, index=False)
    print('stage2_solution.csv updated !')
    print('Images and masks of stage 2 updated !')

def read_image(image_path):
    return imread(image_path).astype(np.uint8)

def binary_mask(id):
    masks_paths = glob(id + '/masks/*.png')
    masks = np.array([read_image(mask_path) for mask_path in masks_paths])
    num_masks, height, width = masks.shape
    binary_mask = np.zeros((height, width), np.uint8)
    for index in range(0, num_masks):
        binary_mask[masks[index] > 0] = 255
    return binary_mask

def load_data_1(train_path):
    train_image_paths = sorted(glob(train_path + '*/images/*.png'))
    train_ids = [path.rsplit('/', 2)[0] for path in train_image_paths]
    train_images = [read_image(image_path) for image_path in tqdm(train_image_paths, desc='Loading train images')]
    train_masks = [binary_mask(id) for id in tqdm(train_ids, desc='Loading train masks')]
    return train_images, train_masks

def load_data_2(train_path, test_1_path, test_2_path):
    train_image_paths = sorted(glob(train_path + '*/images/*.png'))
    test_1_image_paths = sorted(glob(test_1_path + '*/images/*.png'))
    test_2_image_paths = sorted(glob(test_2_path + '*/images/*.png'))

    train_ids = [path.rsplit('/', 2)[0] for path in train_image_paths]
    test_1_ids = [path.rsplit('/', 2)[0] for path in test_1_image_paths]
    test_2_ids = [path.rsplit('/', 2)[0] for path in test_2_image_paths]

    train_images = [read_image(image_path) for image_path in tqdm(train_image_paths, desc='Train images')]
    test_1_images = [read_image(image_path) for image_path in tqdm(test_1_image_paths, desc='Test images (stage 1)')]
    test_2_images = [read_image(image_path) for image_path in tqdm(test_2_image_paths, desc='Test images (stage 2)')]

    train_masks = [binary_mask(id) for id in tqdm(train_ids, desc='Train masks')]
    test_1_masks = [binary_mask(id) for id in tqdm(test_1_ids, desc='Test masks (stage 1)')]
    test_2_masks = [binary_mask(id) for id in tqdm(test_2_ids, desc='Test masks (stage 2)')]

    train_images += test_1_images
    train_masks += test_1_masks

    return train_images, test_1_images, test_2_images, train_masks, test_1_masks, test_2_masks

def load_image_list_from_stage(stage):
    if stage == 'Train':
        train_image_paths = sorted(glob(TRAIN_PATH + '*/images/*.png'))
        image_list = [train_image_paths.split('/')[-1] for train_image_paths in train_image_paths]
    elif stage == 'Stage 1':
        test_1_image_paths = sorted(glob(TEST_1_PATH + '*/images/*.png'))
        image_list = [test_1_image_paths.split('/')[-1] for test_1_image_paths in test_1_image_paths]
    elif stage == 'Stage 2':
        test_2_image_paths = sorted(glob(TEST_2_PATH + '*/images/*.png'))
        image_list = [test_2_image_paths.split('/')[-1] for test_2_image_paths in test_2_image_paths]
    else:
        raise ValueError(f"Invalid stage '{stage}'")

    return image_list

def load_image(stage, index):
    if stage == 'Train':
        train_image_paths = sorted(glob(TRAIN_PATH + '*/images/*.png'))
        image_path = train_image_paths[index]
    elif stage == 'Stage 1':
        test_1_image_paths = sorted(glob(TEST_1_PATH + '*/images/*.png'))
        image_path = test_1_image_paths[index]

    elif stage == 'Stage 2':
        test_2_image_paths = sorted(glob(TEST_2_PATH + '*/images/*.png'))
        image_path = test_2_image_paths[index]
    else:
        raise ValueError(f"Invalid stage '{stage}'")

    return read_image(image_path)

def labeled_mask(id):
    masks_paths = glob(id + '/masks/*.png')
    masks =  np.array([read_image(mask_path) for mask_path in masks_paths])
    masks = np.array([preprocess(mask, MASK_SHAPE, param=True) for mask in masks])
    num_masks, height, width = masks.shape
    labeled_mask = np.zeros((height, width), np.uint16)
    for index in range(0, num_masks):
        labeled_mask[masks[index] > 0] = index + 1
    return labeled_mask

def markers(id):
    masks_paths = glob(id + '/masks/*.png')
    masks = np.array([read_image(mask_path) for mask_path in masks_paths])
    markers = []
    for mask in masks:
        mask = preprocess(mask, MASK_SHAPE, param=True)
        labeled_mask = label(mask)
        regions = regionprops(labeled_mask)
        for region in regions:
            center_row, center_col = region.centroid
            markers.append([center_row, center_col])
    markers = np.array(markers)
    return markers

def load_solution(test_1_path, test_2_path):
    test_1_image_paths = sorted(glob(test_1_path + '*/images/*.png'))
    test_2_image_paths = sorted(glob(test_2_path + '*/images/*.png'))

    test_1_ids = [path.rsplit('/', 2)[0] for path in test_1_image_paths]
    test_2_ids = [path.rsplit('/', 2)[0] for path in test_2_image_paths]

    test_1_labels = [labeled_mask(id) for id in tqdm(test_1_ids, desc='Test labels (stage 1)')]
    test_2_labels = [labeled_mask(id) for id in tqdm(test_2_ids, desc='Test labels (stage 2)')]

    test_1_markers = [markers(id) for id in tqdm(test_1_ids, desc='Test markers (stage 1)')]
    test_2_markers = [markers(id) for id in tqdm(test_2_ids, desc='Test markers (stage 2)')]

    return test_1_labels, test_2_labels, test_1_markers, test_2_markers

def get_model_path(model_name):
    return f'model_{model_name}.keras'

def load_data():
    if not os.path.exists(DATA_PATH):
        unzip_and_structure_data(SOURCE_PATH, UNZIPPED_PATH, DESTINATION_PATH) #Should be run only once
        save_masks(STAGE_1_SOLUTION_PATH, STAGE_1_PATH, 'Stage 1') #Should be run only once
        save_masks(STAGE_2_SOLUTION_PATH, STAGE_2_PATH, 'Stage 2') #Should be run only once
        remove_ignored_images_masks(STAGE_2_SOLUTION_PATH, TEST_2_PATH) #Should be run only once
        print("Data successfully loaded. Can start the app")
    else:
        print("Data already exists. Can start the app")   