🧠 Brain Tumor Classification using Deep Learning

A Deep Learning project that classifies MRI brain images into Tumor or No Tumor, trained using ResNet18, and deployed with a Streamlit web app.

This project includes:

A complete training pipeline in Jupyter Notebook

A Streamlit frontend for real-time predictions

A trained model (best_model.pth) that performs binary classification

Clean project structure for GitHub

📁 Project Structure
Brain-Tumor-Classification/
│
├── Brain_Tumor/                     # Training project (Jupyter Notebook)
│   ├── brain_tumor_classification.ipynb
│   ├── data/
│   │   ├── train/   (empty or sample)
│   │   ├── val/     (empty or sample)
│   │   └── test/    (empty or sample)
│   └── outputs/
│       └── best_model.pth           # trained model (optional)
│
├── Brain_Tumor_App/                 # Streamlit Frontend
│   ├── app.py                       # Web UI
│   ├── model.py                     # Model loading + architecture
│   ├── outputs/
│   │   └── best_model.pth           # model used by Streamlit
│   └── requirements.txt
│
└── README.md
