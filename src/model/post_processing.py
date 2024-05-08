import numpy as np
from skimage.measure import regionprops, label
from tqdm import tqdm
from scipy.ndimage import distance_transform_edt
from skimage.filters import threshold_otsu
from skimage.feature import peak_local_max
from scipy.ndimage import distance_transform_edt, binary_fill_holes
from skimage.morphology import disk
from skimage.segmentation import watershed

def post_proccess_masks(test_pred_masks):
    test_pp_masks = []
    for idx, mask in enumerate(tqdm(test_pred_masks, desc='Post processing of predictions')):
        mask = np.squeeze(mask)
        mask = mask > threshold_otsu(mask)
        mask = binary_fill_holes(mask).astype(np.uint8)
        regions = regionprops(label(mask))
        radii = []
        for region in regions:
            radius = np.sqrt(region.area / np.pi)
            radii.append(radius)
        average_radius = int(round(np.mean(radii)))
        distance = distance_transform_edt(mask)
        peaks = peak_local_max(distance, min_distance=average_radius, footprint=disk(average_radius), threshold_rel=0.2, exclude_border=False)
        peak_mask = np.zeros_like(mask, dtype=bool)
        peak_mask[tuple(peaks.T)] = True
        markers = label(peak_mask)
        wsh = watershed(-distance, markers, mask=mask)
        # markers[mask == 0] = -1
        # rw = random_walker(mask, markers)
        # rw[rw <= 0] = 0
        test_pp_masks.append(wsh)

    return np.array(test_pp_masks)

def post_proccess_masks_2(test_pred_masks, markers):
    test_pp_masks = []
    for idx, mask in enumerate(test_pred_masks):
        mask = np.squeeze(mask)
        distance = distance_transform_edt(mask)
        m2 = markers[idx]
        wsh = watershed(-distance, m2, mask=mask)
        # markers[mask == 0] = -1
        # rw = random_walker(mask, markers)
        # rw[rw <= 0] = 0
        test_pp_masks.append(wsh)

    return np.array(test_pp_masks)
