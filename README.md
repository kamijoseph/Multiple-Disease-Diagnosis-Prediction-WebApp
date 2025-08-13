# 🩺 Multi-Disease Diagnosis WebApp

🔗 **Live Demo**: [Click here to try the deployed app](https://your-deployed-app-link.streamlit.app/)

---

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Built%20with-Streamlit-ff4b4b.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)

An interactive **multi-disease prediction platform** built with **Streamlit**, capable of running **four separate medical risk prediction models** in one interface:

- ❤️ **Heart Disease Predictor**  
- 🧠 **Parkinson’s Disease Predictor**  
- 🩸 **Diabetes Predictor**  
- 🎗 **Breast Cancer Predictor**

Each module uses a trained machine learning model to simulate diagnostic scenarios and show how various medical measurements affect risk classification.

> ⚠️ **Disclaimer**: This application is for **educational and demonstration purposes only**. It is **not a medical diagnostic tool** and must **never replace professional medical advice, diagnosis, or treatment**. Always consult qualified healthcare providers for any medical concerns.

---

## 📁 Project Structure
```bash
MultiDisease-Diagnosis-WebApp/
│
├── app/
│ └── main.py # Main Streamlit multi-page interface
│
├── datasets/
│ ├── diabetes.csv
│ ├── heart.csv
│ ├── breast_cancer.csv
│ └── parkinsons.csv
│
├── models/
│ ├── diabetes_model.sav
│ ├── diabetes_scaler.sav
│ ├── heart_model.sav
│ ├── cancer_model.sav
│ ├── cancer_scaler.sav
│ ├── parkinsons_model.sav
│ └── parkinsons_scaler.sav
│
├── requirements.txt # Required Python libraries
└── README.md
```

---

## 🚀 Features

- 🖥 **Multi-Page Navigation** — Choose from 4 disease predictors in one app.
- 🎯 **Real-Time Predictions** — Instant classification when inputs are provided.
- 📊 **Probability Scores** — Confidence level for each prediction (if model supports it).
- 🧩 **Feature-Specific Inputs** — Customized forms for each disease dataset.
- 🔒 **Scalers Integrated** — Models load with corresponding pre-trained scalers for consistent input preprocessing.
- 📂 **Organized Architecture** — Clear separation of datasets, models, and app UI code.

---

## ⚙️ Installation

### 🔐 Prerequisites
- Python ≥ 3.10
- Conda (recommended)
- Git

---

### 📦 Setup Guide

#### 1. Clone this repository
```bash
git clone https://github.com/your-username/MultiDisease-Diagnosis-WebApp.git
cd MultiDisease-Diagnosis-WebApp
```
#### 2. Create conda environment
```bash
conda create -n multidisease python=3.10
```

#### 3. Activate environment
```bash
conda activate multidisease
```

#### 4. Install dependencies
```bash
conda install --file requirements.txt
```

#### 5. Run Streamlit Application
```bash
cd app
streamlit run main.py
```

---
## Models and Data Sources

| Disease       | Dataset Source                                                                                                      | Model Used                    |
| ------------- | ------------------------------------------------------------------------------------------------------------------- | ----------------------------- |
| Breast Cancer | [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29) | Logistic Regression           |
| Diabetes      | [Kaggle Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)        | Random Forest / SVC           |
| Heart Disease | [UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+Disease)                                  | Random Forest / Logistic Reg. |
| Parkinson’s   | [UCI Parkinson’s Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)                                       | ElasticNet

