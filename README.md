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
9. [Team Contributions](#team-contributions)
10. [Technologies Used](#technologies-used)
11. [License](#license)

---

##  **Project Description**

This project focuses on classifying facial expressions into **seven emotion categories**:(0= 'angry',1= 'disgust', 2= 'fear', 3= 'happy', 4= 'neutral', 5= 'sad', 6= 'surprise')


We developed and compared two models:

- **Baseline CNN** → helped understand dataset behavior but achieved limited accuracy (~74.5%).  
- **Swin Transformer** → significantly improved performance and became the final chosen model.

The project includes:

- Full EDA  
- Preprocessing & Data Augmentation  
- Model training & validation  
- Evaluation metrics  
- GUI deployment using **Streamlit**

---

##  **Dataset**

- **Source:** Facial Emotion Recognition Dataset (Kaggle)  
- **Link:** https://www.kaggle.com/datasets/fahadullaha/facial-emotion-recognition-dataset  
- **Image size:** 96*96 RGB  
- **Train size:** 34844 image (70.0%)
- **Validation size:**  7467 image (15.0%) 
- **Test size:** 7468 image (15.0%)  
- **Dataset notes:**  
  - Faces are centered and pre-aligned  
  - Class distribution is imbalanced (Happy is dominant)

---

##  **Project Pipeline**
Data Loading → Preprocessing → Augmentation → Model Training (CNN/Swin) → Evaluation  → Deployment (Streamlit GUI)

---

##  **Preprocessing**
For CNN
Using `torchvision.transforms`:

python
train_transforms = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


# Validation & Test transforms (without augmentation)
val_test_transforms = transforms.Compose([
    transforms.Resize((96, 96)),
    transforms.Grayscale(num_output_channels=3), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
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

-Achieved ~74.5% accuracy

->The CNN served as a baseline but didn’t generalize well due to dataset complexity.

2️⃣ Swin Transformer (Final Model)

Swin Transformer introduces:

✔ Hierarchical representation

✔ Shifted window mechanism

✔ Strong performance on image classification tasks

✔ Better handling of spatial dependencies

->This model dramatically improved accuracy and per-class metrics.

##  **Training Setup**
For Swin transform

Epochs: 60
Batch size: 64
Optimizer: AdamW
Loss: CrossEntropyLoss
Scheduler: lr_scheduler.CosineAnnealingLR
Early Stopping: patience = 10
Checkpointing: every 5 epochs
Hardware: NVIDIA T4 GPU

## **Results**

<img width="680" height="368" alt="image" src="https://github.com/user-attachments/assets/b38bc09d-277b-43c0-a6eb-d7348ab151f2" />


**Most Confused Emotion Pairs**

<img width="350" height="158" alt="image" src="https://github.com/user-attachments/assets/4b3b0855-20b2-4f93-809e-b93ae2019247" />


 **Per-Class Accuracy & F1:**
 
 <img width="464" height="219" alt="image" src="https://github.com/user-attachments/assets/835f9ec9-4236-4c7e-a9df-902b8a1e94af" />

 
## **Confusion Matrix & Explainability**

We generated:

✔ Confusion Matrix

<img width="935" height="790" alt="image" src="https://github.com/user-attachments/assets/409632d9-8d30-4592-8ca9-48a1113544b5" />



## **Team Contributions**
**Rodina Hesham**

-Led the initial data understanding phase: explored the dataset, analyzed class definitions, and handled dataset loading and structuring.

-Performed all preprocessing steps including normalization, resizing, augmentation, grayscale-to-RGB conversion, and data preparation for model training.

-Conducted data visualization such as class distribution plots and class-weight analysis to understand dataset imbalance.

-Generated sample visualizations to validate the effectiveness of preprocessing steps.

-Developed both the frontend and backend components of the project, ensuring a complete and functional system pipeline from data preprocessing to user interaction.

**Tasneem Yasser**

-Designed, implemented, and optimized the CNN baseline model, establishing the project’s first benchmark and helping the team understand dataset complexity and model behavior.

-Developed a robust training loop including forward pass, backward pass, gradient updates, and performance logging — ensuring stable and reproducible training.

-Configured the loss function, optimizer, and hyperparameters, experimenting with several configurations to improve the initial model’s accuracy and generalization.

-Performed early testing, debugging, and validation of the CNN model pipeline, laying the technical foundation for later advanced models.

-Conducted architectural adjustments and tuning attempts to maximize the CNN model’s performance before moving to transformer-based methods.

**Mai Hussein**

-Led the full evaluation and performance analysis phase by computing key metrics including Accuracy, Precision, Recall, and F1-score for both CNN and Swin Transformer models.

-Generated and interpreted the Confusion Matrix, identifying the most confused emotion pairs and providing insights that guided model improvement.

-Conducted comparative evaluation between baseline and advanced models, highlighting strengths and weaknesses across different emotion categories.

-Prepared detailed analysis reports and supported the understanding of class-wise performance, model fairness, and possible dataset-related issues.

-Contributed to validating the final model’s reliability from an analytical and statistical perspective.



**Maryhan Sabry**

-Implemented and fine-tuned a Swin Transformer model, handling model configuration, pretrained weights, and adapting the architecture to the project’s classification task.

-Designed an advanced training and optimization strategy for Swin, including tuning hyperparameters, stabilizing training, and improving model accuracy.

-Built a complete model inference pipeline, covering preprocessing, forward inference, prediction decoding, and performance handling for large input images.

-Developed a fully interactive Streamlit GUI, enabling users to upload images, run predictions, and view results through a clean and intuitive interface.

-Integrated the Swin Transformer with the Streamlit application, ensuring seamless communication between the model backend and the user-facing frontend.


##  **Technologies Used**

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
