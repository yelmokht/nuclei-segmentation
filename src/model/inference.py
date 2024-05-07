from tqdm import tqdm
import numpy as np
from skimage.filters import threshold_otsu

def tta(model, images):
    tta_masks = []
    for image in tqdm(images, desc='TTA for test images (stage 2)'):
        tta_predictions = []
        original_prediction = model.predict(np.expand_dims(image, axis=0), verbose=0)
        tta_predictions.append(np.squeeze(original_prediction))

        # for angle in range(90, 360, 90):
        #     rotated_image = np.rot90(image, k=angle // 90)
        #     rotated_prediction = model.predict(np.expand_dims(rotated_image, axis=0), verbose=0)
        #     tta_predictions.append(np.rot90(np.squeeze(rotated_prediction), k=-angle // 90))

        # # Horizontal flip
        # horizontal_flip = np.fliplr(image)
        # horizontal_flip_prediction = model.predict(np.expand_dims(horizontal_flip, axis=0), verbose=0)
        # tta_predictions.append(np.fliplr(np.squeeze(horizontal_flip_prediction)))

        # # Vertical flip
        # vertical_flip = np.flipud(image)
        # vertical_flip_prediction = model.predict(np.expand_dims(vertical_flip, axis=0), verbose=0)
        # tta_predictions.append(np.flipud(np.squeeze(vertical_flip_prediction)))

        # Mean of all predictions
        mean_prediction = np.expand_dims(np.mean(tta_predictions, axis=0), axis=-1)

        # Thresholding (example using Otsu's method)
        mean_mask = (mean_prediction > 0.5).astype(np.uint8)

        tta_masks.append(mean_mask)

    return np.array(tta_masks)

