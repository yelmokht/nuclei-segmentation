import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries

def plot_image(image):
    plt.close('all')
    plt.figure(figsize=(4, 4))
    plt.title('Image')
    plt.imshow(image)
    plt.tight_layout()
    fig = plt.gcf()
    return fig

def plot_ground_truth(gt_mask):
    plt.close('all')
    plt.figure(figsize=(4, 4))
    plt.title('Ground truth mask')
    plt.imshow(gt_mask)
    plt.tight_layout()
    fig = plt.gcf()
    return fig

def plot_prediction(image, pp_mask):
    plt.close('all')
    plt.figure(figsize=(4, 4))
    plt.title('Post processed mask')
    plt.imshow(mark_boundaries(image, np.squeeze(pp_mask)))
    plt.tight_layout()
    fig = plt.gcf()
    return fig

def print_plot_history(history):
    plt.close('all')
    fig, axes = plt.subplots(2, 2, figsize=(11, 5))

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

    plt.tight_layout()
    
    return fig