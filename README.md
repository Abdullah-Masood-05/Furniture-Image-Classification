# Furniture Image Classification

This project focuses on classifying grayscale images of furniture into five categories: **beds**, **dinning**, **office**, **plastic**, and **sofa**. The dataset is preprocessed and augmented using OpenCV, and training/testing data is saved in `.npy` format for ease of use in model development.

---

## 📁 Dataset

The dataset should be structured in the following format:

```
dataset/
├── beds/
├── dinning/
├── office/
├── plastic/
└── sofa/
```

Each folder must contain grayscale images (`.jpg`, `.jpeg`, `.png`) representing the corresponding class.

**Dataset Download:**
👉 [Dataset Google Drive Link](#https://drive.google.com/file/d/1XYokiyXlr6dloJP8EhQSwsLdj38e8GWg/view?usp=sharing)

---

## ⚙️ Preprocessing (`preprocess.py`)

This script performs the following tasks:

1. **Train-Test Split**
   Splits data into 70% training and 30% testing (`/dataset/train/` and `/dataset/test/`).

2. **Augmentation (Training Only)**

   * Horizontal Flip
   * Vertical Flip
   * Random Rotation
   * Random Crop & Resize

3. **Deduplication**
   Hashes images to avoid duplicates.

4. **One-Hot Encoding**
   Converts class indices to one-hot encoded vectors.

5. **Label Smoothing**
   Applies smoothing with a factor of `0.1` to help generalization.

6. **Data Saving**
   Saves the following files:

   * `X_train.npy`, `y_train.npy`
   * `X_test.npy`, `y_test.npy`

---

## 📊 Classes

```python
CLASSES = ["beds", "dinning", "office", "plastic", "sofa"]
```

---

## 🧪 Training & Testing

All model training and evaluation should be performed using the `.npy` files saved by `preprocess.py`.

Example:

```python
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")
```

---

## 🔧 Requirements

Install dependencies with:

```bash
pip install numpy opencv-python tqdm
```

---

## 📓 Notebooks

The project includes a Jupyter notebook `main.ipynb`. It is expected to use the preprocessed data for training and evaluating a neural network classifier.

---

## ✅ Notes

* Make sure to run `preprocess.py` before using `main.ipynb`.
* This project is designed to work with **low-resource systems**, focusing on CPU-friendly data sizes.
