import numpy as np
from glob import glob
import os
from model.config import TEST_2_PATH
from tqdm import tqdm
from skimage.transform import resize
import csv
from skimage.morphology import label, erosion, disk
from model.data import read_image

# Run-length encoding taken from https://www.kaggle.com/rakhlin/fast-run-length-encoding-python
def rle_encoding(y):
    dots = np.where(y.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b>prev+1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def pred_to_rles(y):
    for i in range(1, y.max() + 1):
        yield rle_encoding(y == i)

def create_submission(preds, submission_path):
    uuids = [path.rsplit('/', 2)[0] for path in sorted(glob(TEST_2_PATH + '*/images/*.png'))]
    original_shapes = [read_image(image_path).shape[:2] for image_path in sorted(glob(TEST_2_PATH + '*/images/*.png'))]
    with open(submission_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ImageId', 'EncodedPixels'])

        for idx, (uid, y) in tqdm(enumerate(zip(uuids, preds))):
            uid = os.path.basename(uid)
            y = label(erosion(y, footprint=disk(2)))
            y_resized = resize(y, (original_shapes[idx][0], original_shapes[idx][1]), mode='constant', order=0, preserve_range=True, anti_aliasing=True)
            for rle in pred_to_rles(y_resized):
                writer.writerow([uid, ' '.join([str(i) for i in rle])])
