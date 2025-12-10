# 🧠 Diabetic Retinopathy Detection using ResNet50

This project detects the stage of Diabetic Retinopathy (DR) from retinal images using a fine-tuned ResNet50 model. It includes model training, evaluation, and a Gradio-based web interface for predictions.

## 🔍 Features

- Transfer Learning with ResNet50
- Weighted loss to handle class imbalance
- Early stopping and learning rate scheduler
- Gradio app for real-time predictions
- Evaluation using confusion matrix and classification report

## 📚 DR Stages

- Healthy
- Mild DR
- Moderate DR
- Proliferate DR
- Severe DR

## 🏗️ Project Structure

diabetic-retinopathy-detection/
├── models/ # Saved model files (.pth)
├── src/
│ ├── train.py # Training script
│ └── app.py # Gradio app script
├── requirements.txt # Dependency list
└── README.md # This file


## 🚀 Getting Started

### 1. Install Requirements

```bash
pip install -r requirements.txt


2. Train the Model

cd src
python train.py

3. Run the Gradio App
 
cd src
python app.py
> ⚠️ Make sure `diabetic_retinopathy_resnet50_advanced_best.pth` is in the `models/` directory.

📊 Model Evaluation
Confusion matrix

Classification report (precision, recall, F1)

Accuracy/loss graphs over epochs

👤 Author
karthick R
MCA Student 


---

### ✅ Next Step:
Let me know if you're ready to:
> **Push this project (code + `README.md`) to GitHub from VS Code**


