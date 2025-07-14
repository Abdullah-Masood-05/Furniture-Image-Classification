# Furniture Image Classification

**Group Project** I created as a semester project in my 6th semster under supervision of [Dr. Yasir Niaz Khan](https://www.linkedin.com/in/yasirniaz/) along with my fellow  [Bazil Suhail](https://github.com/BazilSuhail). 


[![Open Source Love svg1](https://badges.frapsoft.com/os/v1/open-source.svg?v=103)](#)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat&label=Contributions&colorA=red&colorB=black	)](#)

---

### Overview 
In this project **We** implemented a **pure Artificial Neural Network (ANN)** for furniture image classification, developed entirely from scratch using NumPy and other native libraries from python for calculational purposes. Our goal was to:

1. Design and train a neural network **without high-level frameworks** (like TensorFlow/PyTorch)
2. Classify furniture images into 5 categories:
   - Beds
   - Dining sets
   - Office chairs
   - Plastic furniture  
   - Sofas

Key technical aspects:
- Implemented forward/backpropagation manually
- Used NumPy for all matrix operations
- Focused on core ANN concepts (activation functions, loss calculation, weight updates)


---

### 🤖 Tech Stack 

 <a href="#"> 
<img alt="Python" src="https://img.shields.io/badge/Python-%233776AB.svg?&style=for-the-badge&logo=python&logoColor=white"/>
<img alt="NumPy" src="https://img.shields.io/badge/NumPy-%23013243.svg?&style=for-the-badge&logo=numpy&logoColor=white"/>
<img alt="OpenCV" src="https://img.shields.io/badge/OpenCV-%235C3EE8.svg?&style=for-the-badge&logo=opencv&logoColor=white"/>
<img alt="Neural Networks" src="https://img.shields.io/badge/Neural_Networks-%23FF6F00.svg?&style=for-the-badge&logo=deeplearning&logoColor=white"/>
<img alt="Matplotlib" src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?&style=for-the-badge&logo=matplotlib&logoColor=black"/>
<img alt="tqdm" src="https://img.shields.io/badge/tqdm-%23FFD700.svg?&style=for-the-badge&logo=python&logoColor=blue"/>
 </a>

---

### Key Contributions  
1. **Dataset Preparation**  
   - Collected and labeled furniture images across 5 classes.  
   - Preprocessed and augmented data using **OpenCV** (resizing, normalization, rotations, flips).  
   - Stored training/testing sets in `.npy` format for efficient model loading.  

2. **Model Development**  
   - Experimented with CNN architectures (e.g., ResNet, custom models).  
   - Optimized hyperparameters for accuracy and computational efficiency.  

3. **Collaborative Workflow**  
   - Divided tasks between team members (data preprocessing, model training, evaluation).  
   - Used Git for version control and collaborative coding.  

### Outcome  
Achieved **74.55% accuracy** and gained hands-on experience in end-to-end deep learning pipelines.  

### Tools & Technologies  

- Python, OpenCV
- Data Augmentation, Transfer Learning  
- NumPy for `.npy` data handling  

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
👉 [Dataset Google Drive Link](https://drive.google.com/file/d/1XYokiyXlr6dloJP8EhQSwsLdj38e8GWg/view?usp=sharing)


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
