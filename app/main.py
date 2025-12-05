import streamlit as st
from streamlit_option_menu import option_menu
import os
import pickle 
import numpy as np


# Page configuration

st.set_page_config(
    page_title="Multiple Disease Prediction",
    layout="wide",
    page_icon="🧑‍⚕️"
)

# Load models and scaler

working_dir = os.path.dirname(os.path.abspath(__file__))
diabetes_disease_model = pickle.load(open(f'{working_dir}/model/diabetes_random_model.pkl','rb'))
heart_disease_model = pickle.load(open(f'{working_dir}/model/heart_linear_model.pkl','rb'))
kidney_disease_model = pickle.load(open(f'{working_dir}/model/kidney_linear_model.pkl','rb'))

diabetes_scaler = pickle.load(open(f'{working_dir}/model/diabetes_scaler.pkl','rb'))
kidney_scaler = pickle.load(open(f'{working_dir}/model/kidney_scaler.pkl','rb'))

# Sidebar menu

with st.sidebar:
    selected = option_menu(
        menu_title="Disease Prediction App",
        options=["Diabetes Prediction", "Heart Disease Prediction", "Kidney Disease Prediction"],
        icons=["activity", "heart-pulse", "droplet-half"],
        menu_icon="hospital-fill",
        default_index=0
    )

# Diabetes Prediction Page

if selected == "Diabetes Prediction":
    st.title("🩸 Diabetes Prediction Using Machine Learning")
    st.markdown("#### Enter the details below:")

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.text_input("Pregnancies", placeholder="e.g. 3")
        Glucose = st.text_input("Blood Glucose Level (70 - 320 mg/dL)", placeholder="e.g. 90")
        BloodPressure = st.text_input("Blood Pressure (40 - 180 mmHg)", placeholder="e.g. 60")

    with col2:
        SkinThickness = st.text_input("Skin Thickness (1 - 100 mm)", placeholder="e.g. 25")
        Insulin = st.text_input("Insulin Level (0 - 900 µU/mL)", placeholder="e.g. 120")
        BMI = st.text_input("Body Mass Index (BMI: 5 - 100)", placeholder="e.g. 22")

    with col3:
        DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function (0.1 - 2.5)", placeholder="e.g. 1.2")
        Age = st.text_input("Age (10 - 100)", placeholder="e.g. 21")

    if st.button("Predict Diabetes"):
        try:
            user_input = [float(x) for x in [
                Pregnancies, Glucose, BloodPressure, SkinThickness,
                Insulin, BMI, DiabetesPedigreeFunction, Age
            ]]

            user_input_np = np.array(user_input).reshape(1, -1)
            scaled_input = diabetes_scaler.transform(user_input_np)

            prediction = diabetes_disease_model.predict(scaled_input)

            if prediction[0] == 1:
                diabetes_result = "🩺 The person has diabetes."
            else:
                diabetes_result = "✅ The person does not have diabetes."

            st.success(diabetes_result)

        except ValueError:
            st.error("⚠️ Please enter valid numeric values in all fields.")


# Heart Disease Prediction Page

elif selected == "Heart Disease Prediction":
    st.title("❤️ Heart Disease Prediction")
    st.markdown("#### Enter the patient's details:")

    # Column Layout
    
    col1, col2, col3 = st.columns(3)

    with col1:
        Gender = st.radio("Gender", ["Male (1)", "Female (0)"])
        restecg = st.selectbox("Resting ECG", ["Normal (0)", "ST-T Abnormality (1)", "Left Ventricular Hypertrophy (2)"])
        Age = st.text_input("Age (20 - 100)", placeholder="e.g. 20")
        trestbps = st.text_input("Resting Blood Pressure (trestbps) (90 - 220 mm Hg)", placeholder="e.g. 120")
        thal = st.selectbox("Thal", ["Unknown (0)","Normal (1)", "Fixed Defect (2)", "Reversible Defect (3)"])

    with col2:
        fbs = st.radio("Fasting Blood Sugar", ["Yes (1)", "No (0)"])
        slope = st.selectbox("Slope of ST Segment", ["Upsloping (0)", "Flat (1)", "Downsloping (2)"])
        chol = st.text_input("Cholesterol (120 - 600 mg/dL)", placeholder="e.g. 224") 
        thalach = st.text_input("Max Heart Rate Achieved (thalach) (70 - 220)", placeholder="e.g. 130")

    with col3:
        exang = st.radio("Exercise Induced Angina", ["Yes (1)", "No (0)"])
        cp = st.selectbox("Chest Pain Type", ["Typical Angina (0)", "Atypical Angina (1)", "Non-anginal (2)", "Asymptomatic (3)"])
        ca = st.text_input("Number of Major Vessels (CA) (0-3)", placeholder="e.g. 2")
        oldpeak = st.text_input("Oldpeak (0.0 - 7.0)", placeholder="e.g. 3.1")

    # Prediction Logic
    
    heart_disease_result = ""

    if st.button("Heart Disease Test Result"):
        
        try:
            
            # Convert inputs
            
            Gender = 1 if "Male" in Gender else 0
            fbs = 1 if "Yes" in fbs else 0
            exang = 1 if "Yes" in exang else 0

            cp_map = {
                "Typical Angina": 0, "Atypical Angina": 1,
                "Non-anginal": 2, "Asymptomatic": 3
            }
            restecg_map = {
                "Normal": 0, "ST-T Abnormality": 1,
                "Left Ventricular Hypertrophy": 2
            }
            slope_map = {
                "Upsloping": 0, "Flat": 1, "Downsloping": 2
            }
            thal_map = {
                "Unknown":0, "Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3
            }

            user_input = [
                float(Age),
                Gender,
                cp_map[cp.split(" (")[0]],
                float(trestbps),
                float(chol),
                fbs,
                restecg_map[restecg.split(" (")[0]],
                float(thalach),
                exang,
                float(oldpeak),
                slope_map[slope.split(" (")[0]],
                float(ca),
                thal_map[thal.split(" (")[0]]
            ]

            prediction = heart_disease_model.predict([user_input])

            heart_disease_result = (
                "⚠️ This person is likely to have Not heart disease."
                if prediction[0] == 1 else
                "✅ This person is unlikely to have heart disease."
            )

        except Exception as e:
            st.error(f"Error in input: {e}")

    st.success(heart_disease_result)

# ========== KIDNEY DISEASE PREDICTION ==========

elif selected == "Kidney Disease Prediction":
    st.title("🧪 Kidney Disease Prediction Using Machine Learning")
    st.markdown("#### Enter the details below:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        
        pe = st.radio("Pedal Edema", ["no (0)", "yes (1)"])
        age = st.text_input("Age (2 - 90)", placeholder="e.g. 45")
        bp = st.text_input("Blood Pressure (50 - 180)", placeholder="e.g. 80")
        sg = st.text_input("Specific Gravity (1.005 - 1.025)", placeholder="e.g. 1.015")
        al = st.text_input("Albumin (0 - 5)", placeholder="e.g. 1")
        su = st.text_input("Sugar (0 - 5)", placeholder="e.g. 0")
        rbc = st.selectbox("Red Blood Cells", ["abnormal (0)","normal (1)"])
        pc = st.selectbox("Pus Cell", ["abnormal (0)","normal (1)"])

    with col2:
        
        cad = st.radio("Coronary Artery Disease", ["no (0)", "yes (1)"])
        bgr = st.text_input("Blood Glucose Random (20 - 500)", placeholder="e.g. 121")
        bu = st.text_input("Blood Urea (1.5 - 391)", placeholder="e.g. 44.5")
        sc = st.text_input("Serum Creatinine (0.4 - 76)", placeholder="e.g. 1.2")
        sod = st.text_input("Sodium (4.5 - 163)", placeholder="e.g. 135")
        pot = st.text_input("Potassium (2.5 - 47)", placeholder="e.g. 4.6")
        pcc = st.selectbox("Pus Cell Clumps", ["notpresent (0)", "present (1)"])
        ba = st.selectbox("Bacteria",["notpresent (0)", "present (1)"])

    with col3:
        
        ane = st.radio("Anemia", ["no (0)", "yes (1)"]) 
        hemo = st.text_input("Hemoglobin (3.1 - 17.8)", placeholder="e.g. 13.5")
        pcv = st.text_input("Packed Cell Volume (9 - 54)", placeholder="e.g. 38")
        wc = st.text_input("White Blood Cell Count (2200 - 26400)", placeholder="e.g. 8000")
        rc = st.text_input("Red Blood Cell Count (2.1 - 9.2)", placeholder="e.g. 4.5")
        appet = st.selectbox("Appetite", ["good (0)", "poor (1)"])
        htn = st.selectbox("Hypertension", ["no (0)", "yes (1)"])
        dm = st.selectbox("Diabetes Mellitus", ["no (0)", "yes (1)"])
        
    kidney_result = ""

    if st.button("Predict Kidney Disease"):
        
        try:
                
            pe = 1 if "yes" in pe else 0
            cad = 1 if "yes" in cad else 0
            ane = 1 if "yes" in ane else 0

            rbc_map = { "abnormal":0,"normal":1 }
            pc_map = {"abnormal":0,"normal":1}
            pcc_map = {"notpresent":0,"present":1}
            ba_map = {"notpresent":0,"present":1 }
            appet_map = {"good":0,"poor":1}
            htn_map = {"no":0,"yes":1}
            dm_map = {"no":0,"yes":1}
            
            input_data = [
                float(age), float(bp), float(sg), float(al), float(su),
                rbc_map[rbc.split(" (")[0]], pc_map[pc.split(" (")[0]],
                pcc_map[pcc.split(" (")[0]], ba_map[ba.split(" (")[0]],
                float(bgr), float(bu), float(sc), float(sod), float(pot),
                float(hemo), float(pcv), float(wc), float(rc),
                htn_map[htn.split(" (")[0]], dm_map[dm.split(" (")[0]], cad,
                appet_map[appet.split(" (")[0]], pe, ane
            ]

            input_np = np.array(input_data).reshape(1, -1)
            scaled_input = kidney_scaler.transform(input_np)
            prediction = kidney_disease_model.predict(scaled_input)
            

            if prediction[0] == 1:
                kidney_result = "✅ The person does not have chronic kidney disease."
            else:
                kidney_result = "⚠️ The person has chronic kidney disease."

            st.success(kidney_result)
            
        except Exception as e:
            st.error(f"Unexpected error: {e}")
