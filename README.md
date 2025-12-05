# Multiple Disease Prediction using Machine Learning

<img width="1848" height="613" alt="Capture" src="https://github.com/user-attachments/assets/4cde9018-1ea8-4dcd-8a52-1d6ab11f15ef" />


A Streamlit web application for predicting multiple diseases including **Diabetes**, **Heart Disease**, and **Kidney Disease** using machine learning algorithms. This project provides an intuitive interface for users to input medical information and receive instant disease predictions.

## 📋 Table of Contents
- [Introduction](#-introduction)
- [Features](#-features)
- [Technologies Used](#️-technologies-used)
- [Setup](#-setup)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [License](#-license)

## 🎯 Introduction

The Multiple Disease Prediction project aims to create a user-friendly web application that allows users to input relevant medical information and receive predictions for different diseases. Machine learning models trained on disease-specific datasets enable accurate predictions for diabetes, heart disease, and kidney disease, making health screening more accessible.

## ✨ Features

- **User Input**: Input medical information including age, gender, blood pressure, cholesterol levels, glucose levels, and other relevant factors
- **Disease Prediction**: Utilizes trained ML models to predict the likelihood of diabetes, heart disease, and kidney disease
- **Prediction Results**: Displays predicted outcomes with clear probability indicators
- **Multiple Disease Support**: 
  - Diabetes Prediction
  - Heart Disease Prediction
  - Kidney Disease Prediction
- **User-Friendly Interface**: Intuitive sidebar navigation with clean design
- **Real-time Predictions**: Instant predictions using trained machine learning models
- **No Technical Knowledge Required**: Simple interface for anyone to use

## 🛠️ Technologies Used

- **Python**
- **Streamlit**
- **Streamlit Option Menu**
- **scikit-learn**
- **NumPy**
- **Pandas**
- **Pickle**

## 📦 Setup

Follow these steps to set up the project locally:

### 1. Clone the repository
```bash
git clone https://github.com/zeel0014/Multiple-Disease-Prediction.git
cd Multiple-Disease-Prediction
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
## 🚀 Usage

### Running the Application

1. Open a terminal and navigate to the project directory
2. Navigate to the app folder:
   ```bash
   cd app
   ```
3. Start the Streamlit application:
   ```bash
   streamlit run md.py
   ```
4. Open your web browser and go to `http://localhost:8501`

## 📁 Project Structure

```
Multiple-Disease-Prediction/
│
├── app/
│   ├── md.py                          # Main Streamlit application
│   └── model/
│       ├── diabetes_random_model.pkl  # Diabetes prediction model
│       ├── diabetes_scaler.pkl        # Diabetes data scaler
│       ├── heart_linear_model.pkl     # Heart disease prediction model
│       ├── kidney_linear_model.pkl    # Kidney disease prediction model
│       └── kidney_scaler.pkl          # Kidney data scaler
│
├── dataset/
│   ├── diabetes.csv                   # Diabetes training dataset
│   ├── heart.csv                      # Heart disease training dataset
│   └── kidney_disease.csv             # Kidney disease training dataset
│
├── jupitar_file/
│   ├── Diabetes.ipynb                 # Diabetes model training notebook
│   ├── Heart.ipynb                    # Heart disease model training notebook
│   └── Kidney.ipynb                   # Kidney disease model training notebook
│
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
└── README.md                          # Project documentation
```

## ⚠️ Disclaimer

This application is for educational and informational purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical concerns.

## 📄 License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

**Developed by**: [@zeel0014](https://github.com/zeel0014)  
**Project Link**: [Multiple-Disease-Prediction](https://github.com/zeel0014/Multiple-Disease-Prediction)

---

⭐ **If you find this project helpful, please consider giving it a star!**
