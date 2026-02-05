# Explainable-attention-based-CNN
# Explainable AI-Based Pneumonia Detection

This project implements an **Explainable Attention-Based CNN** for pneumonia detection using chest X-ray images.

## 🔬 Key Concepts Used
- Convolutional Neural Network (CNN)
- Attention Mechanism (CBAM)
- CLAHE Image Enhancement
- Explainable AI (Grad-CAM)

## 📂 Project Structure
Pneumonia/
├── attention_model.py
├── dataset_loader.py
├── preprocessing.py
├── train.py
├── predict_single_image.py
├── gradcam_single_image.py


## 🧪 Dataset
Dataset used: Kaggle Chest X-ray Pneumonia  
(Not uploaded due to size restrictions)

## ▶️ How to Run
```bash
pip install -r requirements.txt
python train.py
python predict_single_image.py
python gradcam_single_image.py
