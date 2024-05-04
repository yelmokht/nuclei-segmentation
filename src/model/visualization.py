import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries
# from ipywidgets import interact, IntSlider, Layout
# from matplotlib.colors import LinearSegmentedColormap

# def plot_images(train_images, train_masks, test_1_images, test_1_masks, test_2_images, test_2_masks):
#     @interact(index_0=IntSlider(min=0, max=len(train_images)-1, continuous_update=False, description='Train index'),
#               index_1=IntSlider(min=0, max=len(test_1_images)-1, continuous_update=False, description='Test 1 index'),
#               index_2=IntSlider(min=0, max=len(test_2_images)-1, continuous_update=False, description='Test 2 index'))
#     def update_plot(index_0, index_1, index_2):
#         fig, axes = plt.subplots(2, 3, figsize=(18, 8), layout='compressed')

#         axes[0, 0].imshow(train_images[index_0])
#         axes[0, 0].set_title(f'Train image n°{index_0}')
#         axes[1, 0].imshow(train_masks[index_0])
#         axes[1, 0].set_title(f'Train mask n°{index_0}')

#         axes[0, 1].imshow(test_1_images[index_1])
#         axes[0, 1].set_title(f'Test 1 image n°{index_1}')
#         axes[1, 1].imshow(test_1_masks[index_1])
#         axes[1, 1].set_title(f'Test 1 mask n°{index_1}')

#         axes[0, 2].imshow(test_2_images[index_2])
#         axes[0, 2].set_title(f'Test 2 image n°{index_2}')
#         axes[1, 2].imshow(test_2_masks[index_2])
#         axes[1, 2].set_title(f'Test 2 mask n°{index_2}')

#         plt.show()

def plot_image(image):
    plt.close('all')
    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    fig = plt.gcf()
    return fig

# def visualize_augmented_data(train_generator, val_generator, steps_per_epoch, validation_steps):
#     batch_size = BATCH_SIZE
#     num_cols = 4

#     @interact(batch_num=IntSlider(min=0, max=min(steps_per_epoch - 1, validation_steps - 1), continuous_update=False, description='Batch number'))
#     def update_plot(batch_num):
#         fig, axes = plt.subplots(8, 8, figsize=(32, 32))


#         for i, (train_images_batch, _) in enumerate(train_generator):
#             if i == batch_num:
#                 break

#         for i, (val_images_batch, _) in enumerate(val_generator):
#             if i == batch_num:
#                 break

#         for i in range(batch_size):
#             row = i // num_cols
#             col = i % num_cols
#             axes[row, col].imshow(train_images_batch[i])
#             axes[row, col].set_title(f'Train image n°{i}')

#             axes[row, col + num_cols].imshow(val_images_batch[i])
#             axes[row, col + num_cols].set_title(f'Val image n°{i}')

#         plt.show()

def print_plot_history(history):
    plt.close('all')
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout='compressed')

    axes[0, 0].plot(history['iou_score'], label='Train Iou')
    axes[0, 0].plot(history['loss'], label='Train Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_xlabel('Epochs')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].legend()

    axes[0, 1].plot(history['val_iou_score'], label='Val Iou')
    axes[0, 1].plot(history['val_loss'], label='Val Loss')
    axes[0, 1].set_title('Validation Loss')
    axes[0, 1].set_xlabel('Epochs')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].legend()

    axes[1, 0].plot(history['accuracy'], label='Train Accuracy')
    axes[1, 0].plot(history['precision'], label='Train Precision')
    axes[1, 0].plot(history['recall'], label='Train Recall')
    axes[1, 0].plot(history['f1-score'], label='Train F1 Score')
    axes[1, 0].set_title('Training Metrics')
    axes[1, 0].set_xlabel('Epochs')
    axes[1, 0].set_ylabel('Metrics')
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].legend()

    axes[1, 1].plot(history['val_accuracy'], label='Val Accuracy')
    axes[1, 1].plot(history['val_precision'], label='Val Precision')
    axes[1, 1].plot(history['val_recall'], label='Val Recall')
    axes[1, 1].plot(history['val_f1-score'], label='Val F1 Score')
    axes[1, 1].set_title('Validation Metrics')
    axes[1, 1].set_xlabel('Epochs')
    axes[1, 1].set_ylabel('Metrics')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].legend()
    
    return fig


def plot_prediction(image, pp_mask):
    plt.close('all')
    plt.figure(figsize=(4, 4))
    plt.imshow(mark_boundaries(image, np.squeeze(pp_mask)))
    fig = plt.gcf()
    return fig


# def show_results(images, gt_masks, labels, pred_masks, pp_masks, verbose=True):
#     colors = [(0, 0, 1), (1, 1, 1), (1, 0, 0)]  # (blue, white, red)
#     cmap_name = 'custom_cmap'
#     cm = LinearSegmentedColormap.from_list(cmap_name, colors)

#     scores = []
#     for idx, (image, gt_mask, pred_mask, pp_mask) in enumerate(zip(images, gt_masks, pred_masks, pp_masks)):
#         if verbose:
#             fig, ax = plt.subplots(1, 4, figsize=(20, 20))

#             ax[0].set_title(f'Image n°{idx}')
#             ax[0].imshow(image)

#             ax[1].set_title(f'Ground truth mask n°{idx}')
#             ax[1].imshow(gt_mask)

#             ax[2].set_title(f'Post processed mask n°{idx}')
#             ax[2].imshow(np.expand_dims(pp_mask, axis=-1))

#             a = np.squeeze(gt_masks[idx]) - (pp_masks[idx] > 0).astype(np.uint8)
#             ax[3].set_title('Differences')
#             ax[3].imshow(a, cmap=cm, vmin=-1, vmax=1)

#         a = labels[idx]
#         b = label(pp_mask)
#         score_i = score(a, b, verbose=verbose)
#         scores.append(score_i)
#         plt.show()

#     print(f'LB = {np.mean(scores)}')

# visualize_augmented_data(train_generator, val_generator, steps_per_epoch, validation_steps)
# plot_images(train_images, train_masks, test_1_images, test_1_masks, test_2_images, test_2_masks)
# plot_images(pp_train_images, pp_train_masks, pp_test_1_images, pp_test_1_masks, pp_test_2_images, pp_test_2_masks)
# print_plot_history(history)
# # show_results(test_1_images, test_1_masks, test_1_labels, test_1_pred_masks, test_1_pp_masks, verbose=False)
