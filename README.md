# Pneumonia-Detection-using-CNN
# Pneumonia Detection Using CNN

##  Project Overview
This project utilizes **Convolutional Neural Networks (CNNs)** to classify **chest X-ray images** as either **Pneumonia** or **Normal**. The goal is to develop an **automated and accurate diagnostic system** that assists medical professionals in detecting pneumonia more efficiently.

##  Dataset Information
- The dataset consists of **5,863 chest X-ray images** categorized into two classes:
  - **Pneumonia**
  - **Normal**
- The dataset is divided into **training, validation, and test sets**.
- Images are in **JPEG format** and have been medically screened for accuracy.

##  Model Architecture
This deep learning model is developed using **TensorFlow** and **Keras** with the following structure:
- **Convolutional Layers** – Feature extraction with ReLU activation.
- **Max Pooling Layers** – Reducing dimensionality while preserving key features.
- **Fully Connected Dense Layers** – Learning high-level patterns.
- **Dropout Layers** – Preventing overfitting.
- **Softmax Activation** – Binary classification of pneumonia vs. normal cases.

##  Training Pipeline
### **1️ Data Preprocessing**
 Image resizing and normalization.
 Data augmentation to enhance model generalization.

### **2️ Model Training**
 **Loss Function:** Categorical Cross-Entropy.
 **Optimizer:** Adam (Learning Rate = 0.001).
 **Early Stopping:** Prevents overfitting and improves model efficiency.

### ** Model Evaluation**
- Model tested on **unseen test data**.
- Evaluation metrics include:
  - **Accuracy**
  - **Precision & Recall**
  - **F1-score**
  - **Confusion Matrix for Misclassification Analysis**

##  Results & Performance Metrics
- **Training Accuracy:** 95.64%
- **Validation Accuracy:** 87.50%
- **Loss and Accuracy Trends:** Visualized using graphs.
- **Confusion Matrix:** Identifies classification errors and improvements.

## 🛠 Installation & Usage
### **1️ Clone the Repository**
```bash
git clone https://github.com/Nimishachandran/Pneumonia-Detection-using-cnn.git
cd pneumonia-detection-cnn
```

### ** Install Dependencies**
```bash
pip install -r requirements.txt
```

### ** Run the Project**
Execute the model training and testing script:
```bash
python pneumonia_detection.py
```
Or, open the Jupyter Notebook for interactive execution:
```bash
jupyter notebook Pneumonia_Detection.ipynb
```

##  Folder Structure
```
pneumonia-detection-cnn/
│── Pneumonia_Detection.ipynb  # Jupyter Notebook for training & evaluation
│── README.md                   # Project Documentation
│── requirements.txt            # Dependencies
│── model/                      # Saved trained models
│── dataset/                    # Chest X-ray dataset
│── results/                     # Performance metrics & plots
```

## References
- Dataset Source: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

##  License
This project is for **educational and research purposes only**.

---
Developed as part of a **Deep Learning Project for Medical Image Classification**.

