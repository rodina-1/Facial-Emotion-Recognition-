# Facial Emotion Recognition (FER) using Swin Transformer

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Status](https://img.shields.io/badge/Status-Completed-success.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

Facial Emotion Recognition (FER) is a computer vision task that aims to classify human facial expressions into predefined emotion categories.  
This project compares **basic CNN models** with an advanced **Swin Transformer** architecture, showing a significant improvement in accuracy and robustness.  
The final deployed model is based on **Swin Transformer**, due to its superior performance on the RAF-DB dataset.

---

##  **Table of Contents**
1. [Project Description](#project-description)  
2. [Dataset](#dataset)  
3. [Project Pipeline](#project-pipeline)  
4. [Preprocessing](#preprocessing)  
5. [Model Architectures](#model-architectures)  
6. [Training Setup](#training-setup)  
7. [Results](#results)  
8. [Confusion Matrix & Explainability](#confusion-matrix--explainability)  
9. [How to Run](#how-to-run)  
10. [Project Structure](#project-structure)  
11. [Team Contributions](#team-contributions)  
12. [Technologies](#technologies)  
13. [License](#license)

---

##  **Project Description**

This project focuses on classifying facial expressions into **seven emotion categories**:(0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral)


We developed and compared two models:

- **Baseline CNN** → helped understand dataset behavior but achieved limited accuracy (~62%).  
- **Swin Transformer** → significantly improved performance and became the final chosen model.

The project includes:

- Full EDA  
- Preprocessing & Data Augmentation  
- Model training & validation  
- Evaluation metrics  
- Explainability using **Grad-CAM**  
- GUI deployment using **Streamlit**

---

##  **Dataset**

- **Source:** RAF-DB (Kaggle)  
- **Link:** https://www.kaggle.com/datasets/msambare/fer2013  
- **Image size:** 48×48 grayscale  
- **Train size:** 12,271 images  
- **Test size:** 3,068 images  
- **Dataset notes:**  
  - Faces are centered and pre-aligned  
  - Class distribution is imbalanced (Happy is dominant)

---

##  **Project Pipeline**
Data Loading → Preprocessing → Augmentation → Model Training (CNN/Swin) → Evaluation → Grad-CAM → Deployment (Streamlit GUI)

---

##  **Preprocessing**

Using `torchvision.transforms`:

python
train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

test_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((48, 48)),
    ToTensor(),
    Normalize([0.5], [0.5])
])
✔ Normalization

✔ Data Augmentation

✔ Grayscale conversion

✔ Image resizing

##  **Model Architectures**
1️⃣ Baseline CNN (Initial Experiment)

-Simple convolutional layers

-ReLU activation

-MaxPooling

-Fully connected head

-Achieved ~62% accuracy

->The CNN served as a baseline but didn’t generalize well due to dataset complexity.

2️⃣ Swin Transformer (Final Model)

Swin Transformer introduces:
✔ Hierarchical representation

✔ Shifted window mechanism

✔ Strong performance on image classification tasks

✔ Better handling of spatial dependencies

->This model dramatically improved accuracy and per-class metrics.

##  **Training Setup**

Epochs: 60
Batch size: 64
Optimizer: Adam
Loss: CrossEntropyLoss
Scheduler: (if used)
Early Stopping: patience = 10
Checkpointing: every 5 epochs
Hardware: NVIDIA T4 GPU

## **Results**

<img width="603" height="448" alt="image" src="https://github.com/user-attachments/assets/411324c9-d4a4-4010-995d-a916c03e41de" />

Most Confused Emotion Pairs

Happy → Neutral : 47

Neutral → Sad : 46

Sad → Neutral : 38

Disgust → Neutral : 20

## **Confusion Matrix & Explainability**

We generated:

✔ Confusion Matrix

<img width="935" height="790" alt="image" src="https://github.com/user-attachments/assets/1e63d4ac-5d04-46ed-9ca5-5c13e30d3ae6" />


✔ Grad-CAM heatmaps

→ Showing which facial regions the model focuses on when predicting emotions.

This enhances model interpretability and supports academic discussion.

## **Team Contributions**
**Rodina Hesham**

-Data Cleaning

-Preprocessing

-Augmentation

-Dataset Splitting

-Exploratory Data Analysis


**Tasneem Yasser**

-CNN model building

-Swin Transformer fine-tuning

-Training loop

-Loss & optimizer configuration

**Mai Hussein**
Evaluation metrics

-Accuracy

-Precision

-Recall

-F1-score

-Confusion Matrix

-Grad-CAM explainability

**Maryhan Sabry**

-Deployment
-Streamlit GUI
-User interface
-Model inference pipeline

## **Technologies Used**

-Python

-PyTorch

-Torchvision

-Matplotlib

-Streamlit

-Scikit-learn

-seaborn

-Jupyter Notebook

-GPU: NVIDIA T4


📄 License

This project is released under the MIT License and is free for academic and research use.
