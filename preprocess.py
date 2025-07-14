import os
import cv2
import numpy as np
import random
import hashlib
from tqdm import tqdm
import shutil

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Hyperparameters
IMG_SIZE = 128
CLASSES = ["beds", "dinning", "office", "plastic", "sofa"]
BASE_PATH = "dataset"


def split_dataset(base_path, classes, train_ratio=0.7):
    train_path = os.path.join(base_path, "train")
    test_path = os.path.join(base_path, "test")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print("Train and test folders already exist. Skipping split.")
        return

    print("Splitting dataset into train and test folders...")
    for cls in classes:
        class_path = os.path.join(base_path, cls)
        if not os.path.exists(class_path):
            print(f"Warning: Class directory {class_path} does not exist!")
            continue

        train_class_path = os.path.join(train_path, cls)
        test_class_path = os.path.join(test_path, cls)

        os.makedirs(train_class_path, exist_ok=True)
        os.makedirs(test_class_path, exist_ok=True)

        images = [
            f
            for f in os.listdir(class_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        if not images:
            print(f"Warning: No images found in {class_path}")
            continue

        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)

        for i, img in enumerate(images):
            src = os.path.join(class_path, img)
            dst = os.path.join(
                train_class_path if i < split_idx else test_class_path, img
            )
            shutil.copy(src, dst)

    print(
        f"Dataset split completed: {train_ratio*100}% train, {(1-train_ratio)*100}% test."
    )


def get_image_hash(img_array):
    return hashlib.sha256(img_array.tobytes()).hexdigest()


def horizontal_flip(img):
    return cv2.flip(img, 1)


def vertical_flip(img):
    return cv2.flip(img, 0)


def rotate(img):
    angles = [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]
    angle = random.choice(angles)
    return cv2.rotate(img, angle)


def random_crop_resize(img, crop_ratio=0.8):
    h, w = img.shape
    crop_size = int(min(h, w) * crop_ratio)
    x = random.randint(0, w - crop_size)
    y = random.randint(0, h - crop_size)
    cropped = img[y : y + crop_size, x : x + crop_size]
    resized = cv2.resize(cropped, (w, h))
    return resized


def preprocess_images(path, img_size=128, augment=True):
    images = []
    labels = []
    seen_hashes = set()
    augmentation_functions = [
        horizontal_flip,
        vertical_flip,
        rotate,
        random_crop_resize,
    ]

    for label, category in enumerate(CLASSES):
        category_path = os.path.join(path, category)
        if not os.path.exists(category_path):
            print(f"Warning: {category_path} does not exist!")
            continue

        print(f"Processing category: {category}")
        for img_name in os.listdir(category_path):
            if not img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif")):
                continue

            img_path = os.path.join(category_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Failed to read: {img_path}")
                continue

            img_resized = cv2.resize(img, (img_size, img_size))
            img_hash = get_image_hash(img_resized)

            if img_hash in seen_hashes:
                continue
            seen_hashes.add(img_hash)

            # Add original image
            images.append(img_resized.flatten() / 255.0)
            labels.append(label)

            if augment:
                # Randomly select two augmentations
                selected_augs = random.sample(augmentation_functions, 2)
                for aug in selected_augs:
                    augmented_img = aug(img_resized)
                    images.append(augmented_img.flatten() / 255.0)
                    labels.append(label)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels)

    return images, labels


def one_hot_encode(labels, num_classes):
    m = len(labels)
    encoded = np.zeros((m, num_classes))
    encoded[np.arange(m), labels] = 1
    return encoded


def smooth_labels(y, smooth_factor=0.1):
    y *= 1 - smooth_factor
    y += smooth_factor / y.shape[1]
    return y


# Split the dataset
split_dataset(BASE_PATH, CLASSES)

# Define paths
train_path = os.path.join(BASE_PATH, "train")
test_path = os.path.join(BASE_PATH, "test")

# Preprocess training data
print("Loading and preprocessing training data...")
X_train, y_train_indices = preprocess_images(train_path, IMG_SIZE, augment=True)

# Check for data imbalance
class_counts = np.bincount(y_train_indices, minlength=len(CLASSES))
print("Class counts:", class_counts)
if np.max(class_counts) / np.min(class_counts) > 1.5:
    print("Data imbalance detected. Applying oversampling...")
    max_count = np.max(class_counts)
    for i in range(len(CLASSES)):
        if class_counts[i] < max_count:
            extra = max_count - class_counts[i]
            indices_i = np.where(y_train_indices == i)[0]
            selected_indices = np.random.choice(indices_i, extra, replace=True)
            X_train = np.vstack((X_train, X_train[selected_indices]))
            y_train_indices = np.hstack(
                (y_train_indices, y_train_indices[selected_indices])
            )

# Shuffle the training data
indices = np.arange(len(X_train))
np.random.shuffle(indices)
X_train = X_train[indices]
y_train_indices = y_train_indices[indices]

# One-hot encode and smooth labels
y_train = one_hot_encode(y_train_indices, len(CLASSES))
y_train = smooth_labels(y_train)

# Preprocess test data
print("Loading and preprocessing test data...")
X_test, y_test_indices = preprocess_images(test_path, IMG_SIZE, augment=False)
y_test = one_hot_encode(y_test_indices, len(CLASSES))

# Save preprocessed data
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("Preprocessing completed and data saved.")