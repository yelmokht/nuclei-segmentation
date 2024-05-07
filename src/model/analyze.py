import numpy as np
import matplotlib.pyplot as plt
from collections import Counter

def analyze_data(train_images, test_1_images, test_2_images):
    train_sizes = [(image.shape[0], image.shape[1]) for image in train_images]
    test_1_sizes = [(image.shape[0], image.shape[1]) for image in test_1_images]
    test_2_sizes = [(image.shape[0], image.shape[1]) for image in test_2_images]

    all_sizes = train_sizes + test_1_sizes + test_2_sizes
    unique_sizes, counts = np.unique(all_sizes, return_counts=True, axis=0)
    unique_sizes = [tuple(size) for size in unique_sizes]

    fig, ax = plt.subplots(figsize=(10, 6))

    for label, sizes in zip(['Train', 'Test 1', 'Test 2'], [train_sizes, test_1_sizes, test_2_sizes]):
        sizes_flat = np.array(sizes)
        sizes_tuples = [tuple(size) for size in sizes_flat]
        counts = Counter(sizes_tuples)
        ax.bar(range(len(unique_sizes)), [counts[size_tuple] for size_tuple in unique_sizes], alpha=0.5, label=label)

    ax.set_title('Image Sizes Distribution')
    ax.set_xlabel('Image size (height x width)')
    ax.set_ylabel('Count')
    ax.set_xticks(range(len(unique_sizes)))
    ax.set_xticklabels([f"{size[0]}x{size[1]}" for size in unique_sizes], rotation=45)
    ax.legend()

    plt.tight_layout()
    plt.show()

    most_common_overall = Counter(tuple(map(tuple, all_sizes))).most_common(1)[0][0]

    most_common = {
        "Train set": Counter(tuple(map(tuple, train_sizes))).most_common(1)[0][0],
        "Test 1 set": Counter(tuple(map(tuple, test_1_sizes))).most_common(1)[0][0],
        "Test 2 set": Counter(tuple(map(tuple, test_2_sizes))).most_common(1)[0][0],
        "Overall": most_common_overall
    }

    for set_name, size in most_common.items():
        print(f"Most common size in {set_name}:", size)

    return most_common_overall