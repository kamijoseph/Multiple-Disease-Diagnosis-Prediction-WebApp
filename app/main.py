
# streamlit web applicatioon
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np
import plotly.graph_objects as go
import pandas as pd

# importing the models and scalers
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

# the models
diabetes_model = load_pickle("models/diabetes_model.sav")
diabetes_scaler = load_pickle("models/diabetes_scaler.sav")
heart_model = load_pickle("models/heart_model.sav")
parkinsons_model = load_pickle("models/parkinsons_model.sav")
parkinsons_scaler = load_pickle("models/parkinsons_scaler.sav")
breast_cancer_model = load_pickle("models/cancer_model.sav")
breast_cancer_scaler = load_pickle("models/cancer_scaler.sav")

# breast cancer predictive system
def breast_cancer():
    #data cleaning function
    def breast_cleaned_data():
        #importing the dataset
        data = pd.read_csv("datasets/breast_cancer.csv")

        # dropping unnamed: 32 column and the id column
        data = data.drop(["Unnamed: 32", "id"], axis=1)

        # encoding categorical variable: {diagnosis} into numerical values
        data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})

        return data

    # sidebar function
    def add_sidebar():

        # creating the sidebar
        st.sidebar.header("Nuclei Values")

        # importing and cleaning the data
        data = breast_cleaned_data()

        # initializing the slider labels
        slider_labels = [
            ("Radius (mean)", "radius_mean"),
            ("Texture (mean)", "texture_mean"),
            ("Perimeter (mean)", "perimeter_mean"),
            ("Area (mean)", "area_mean"),
            ("Smoothness (mean)", "smoothness_mean"),
            ("Compactness (mean)", "compactness_mean"),
            ("Concavity (mean)", "concavity_mean"),
            ("Concave points (mean)", "concave points_mean"),
            ("Symmetry (mean)", "symmetry_mean"),
            ("Fractal dimension (mean)", "fractal_dimension_mean"),
            ("Radius (se)", "radius_se"),
            ("Texture (se)", "texture_se"),
            ("Perimeter (se)", "perimeter_se"),
            ("Area (se)", "area_se"),
            ("Smoothness (se)", "smoothness_se"),
            ("Compactness (se)", "compactness_se"),
            ("Concavity (se)", "concavity_se"),
            ("Concave points (se)", "concave points_se"),
            ("Symmetry (se)", "symmetry_se"),
            ("Fractal dimension (se)", "fractal_dimension_se"),
            ("Radius (worst)", "radius_worst"),
            ("Texture (worst)", "texture_worst"),
            ("Perimeter (worst)", "perimeter_worst"),
            ("Area (worst)", "area_worst"),
            ("Smoothness (worst)", "smoothness_worst"),
            ("Compactness (worst)", "compactness_worst"),
            ("Concavity (worst)", "concavity_worst"),
            ("Concave points (worst)", "concave points_worst"),
            ("Symmetry (worst)", "symmetry_worst"),
            ("Fractal dimension (worst)", "fractal_dimension_worst"),
        ]

        # empty input dictionary
        input_dict = {}

        # creating the sliders one by one
        for  label, key in slider_labels:

            # minimum , mean and maximum values of the sliders
            min_val = float(0)
            max_val = float(data[key].max())
            mean_val = float(data[key].mean())

            # sliders
            input_dict[key] = st.sidebar.slider(
                label,
                min_value=min_val,
                max_value=max_val,
                value=mean_val
            )
        
        # return the populated input dictionary
        return input_dict

    # scaler function to scale the input for the radar chart
    def breast_scaled_values(input_dict):
        data = breast_cleaned_data()
        X= data.drop(["diagnosis"], axis=1)
        scaled_dict = {}

        for key, value in input_dict.items():
            max_val = X[key].max()
            min_val = X[key].min()
            scaled_value = (value - min_val) / (max_val - min_val)
            scaled_dict[key] = scaled_value

        return scaled_dict

    # radar chart function
    def build_radar_chart(input_data):

        input_data = breast_scaled_values(input_data)

        categories = [
            'Radar', 'Texture', 'Perimeter', 'Area', 'Smoothness',
            'Compactness', 'Concavity', 'Concave Points',
            'Symmetry', 'Fractal Dimension'
        ]

        fig = go.Figure()

        fig.add_trace(go.Scatterpolar(
            r=[
                input_data['radius_mean'],
                input_data['texture_mean'],
                input_data['perimeter_mean'],
                input_data['area_mean'],
                input_data['compactness_mean'],
                input_data['concavity_mean'],
                input_data['concave points_mean'],
                input_data['symmetry_mean'],
                input_data['fractal_dimension_mean']
            ],
            theta=categories,
            fill='toself',
            name='Mean Value'
        ))
        fig.add_trace(go.Scatterpolar(
            r=[
                input_data['radius_se'],
                input_data['texture_se'],
                input_data['perimeter_se'],
                input_data['area_se'],
                input_data['compactness_se'],
                input_data['concavity_se'],
                input_data['concave points_se'],
                input_data['symmetry_se'],
                input_data['fractal_dimension_se']
            ],
            theta=categories,
            fill='toself',
            name='Standard Error'
        ))
        fig.add_trace(go.Scatterpolar(
            r=[
                input_data['radius_worst'],
                input_data['texture_worst'],
                input_data['perimeter_worst'],
                input_data['area_worst'],
                input_data['compactness_worst'],
                input_data['concavity_worst'],
                input_data['concave points_worst'],
                input_data['symmetry_worst'],
                input_data['fractal_dimension_worst']
            ],
            theta=categories,
            fill='toself',
            name='Worst Value'
        ))
        fig.update_layout(
            width=800,
            height=700,
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True
        )
        
        return fig

    # prediction column functiom
    def predictions_column(input_data):

        # converting input data to a numpy array
        input_asarray = np.array(list(input_data.values()))

        # reshaping the input array to 2d
        input_reshaped = input_asarray.reshape(1, -1)

        # scaling the input data using the imported scaler
        input_scaled = breast_cancer_scaler.transform(input_reshaped)

        # Prediction using the imported model
        prediction = breast_cancer_model.predict(input_scaled)

        st.subheader("Cell Cluster Prediction")
        st.divider()
        st.write("The cell cluster is:")

        # checking if prediction is malignant or benign
        if (prediction[0] == 1):
            st.write("**MALIGNANT TUMOUR**")
        else:
            st.write("**BENIGN TUMOUR**")

        # probability for both classes
        benign_prob = breast_cancer_model.predict_proba(input_scaled)[0][0]
        malignant_prob = breast_cancer_model.predict_proba(input_scaled)[0][1]

        st.divider()

        # write the probabilities to the interface
        st.write("Benign Probability:", benign_prob)
        st.write(f"Malignant Probability:", malignant_prob)

        st.divider()
        
        # Awareness
        st.write("This project is to be used by medical professional in making a diagnosis but is never a substitute for professional and proper medical diagnosis")
    
    # main function
    st.set_page_config(
        page_title="Breast Cancer Predictor",
        page_icon=":male-doctor:",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # creating input variable by calling the sidebar fnction to create it
    input_data = add_sidebar()

    # title and introduction container
    with st.container():
        st.title("Breast Cancer Prediction Model")
        st.write("The prediction model works by taking different inputs through the slider or by connecting your cytosis lab results. The model classifies breast lumps, mass or tumour into either Malignant or Benign with Malignant being cancerous tumour and the opposite for Benign")
    
    # columns initialization
    col1, col2 = st.columns([4, 1], border=True )

    # columns application
    with col1:
        radar_chart = build_radar_chart(input_data)
        st.plotly_chart(radar_chart)
    with col2:
        predictions_column(input_data)

# diabetes predictive system
def diabetes():
    # diabetes predictor function
    def diabetes_predictor(input_data):
        # converting to array
        input_array = np.asarray(input_data)

        # reshaping the input array
        input_reshaped = input_array.reshape(1, -1)

        # scaling the reshaped input
        input_scaled = diabetes_scaler.transform(input_reshaped)

        # model prediction
        prediction = diabetes_model.predict(input_scaled)[0]

        # class probabilities
        probabilities = diabetes_model.predict_proba(input_scaled)[0]

        # get results and class probabilities and return them
        result = "DIABETIC!, Seek medical help." if prediction == 1 else "NOT DIABETIC, but still seek medical diagnosis."
        prob_dict = {
            "Not Diabetic": round(probabilities[0] * 100, 2),
            "Diabetic": round(probabilities[1] * 100, 2)
        }
        return result, prob_dict

    # title
    st.title("Diabetes Prediction Model")
    st.write("This project is to be used by medical professional in making a diagnosis but is never a substitute for professional and proper medical diagnosis")
    st.write("The prediction model works by taking different inputs through the slider on the left or by manually filling in the values in the prompt. Press get results when you are done.")

    # Sidebar sliders
    st.sidebar.header("Diabetes Sliders")

    preg_slider = st.sidebar.slider("Pregnancies", 0, 20, value=3, step=1)
    glu_slider = st.sidebar.slider("Glucose", 0, 200, value=128, step=1)
    bp_slider = st.sidebar.slider("Blood Pressure", 0, 130, value=62, step=1)
    skin_slider = st.sidebar.slider("Skin Thickness", 0, 100, value=24, step=1)
    insulin_slider = st.sidebar.slider("Insulin", 0, 300, value=52, step=1)
    bmi_slider = st.sidebar.slider("BMI", 0.0, 70.0, value=30.9, step=0.1)
    dpf_slider = st.sidebar.slider("Diabetes Pedigree Function", 0.0, 3.0, value=0.82, step=0.01)
    age_slider = st.sidebar.slider("Age", 10, 100, value=33, step=1)

    # getting input data from the user
    Pregnancies = st.text_input("Number of Pregnancies", value=str(preg_slider))
    Glucose = st.text_input("Glucose Level", value=str(glu_slider))
    BloodPressure = st.text_input("Blood Pressure", value=str(bp_slider))
    SkinThickness = st.text_input("Skin Thickness", value=str(skin_slider))
    Insulin = st.text_input("Insulin Level", value=str(insulin_slider))
    BMI = st.text_input("BMI", value=str(bmi_slider))
    DiabetesPedigreeFunction = st.text_input("Diabetes Pedigree Function", value=str(dpf_slider))
    Age = st.text_input("Age", value=str(age_slider))

    # Final values: if user inputs text, use it; else fallback to slider
    def get_final_value(text_value, slider_value, is_float=False):
        try:
            return float(text_value) if is_float else int(text_value)
        except:
            return slider_value

    final_input = [
        get_final_value(Pregnancies, preg_slider),
        get_final_value(Glucose, glu_slider),
        get_final_value(BloodPressure, bp_slider),
        get_final_value(SkinThickness, skin_slider),
        get_final_value(Insulin, insulin_slider),
        get_final_value(BMI, bmi_slider, is_float=True),
        get_final_value(DiabetesPedigreeFunction, dpf_slider, is_float=True),
        get_final_value(Age, age_slider),
    ]

    # rendering prediction
    diagnosis = ""

    # get results and output probabilities
    if st.button("Get Results"):
        result, prob_dict = diabetes_predictor(final_input)
        st.success(result)
        for cls, prob in prob_dict.items():
            st.write(f"- **{cls}**: {prob}%")
    
    st.success(diagnosis)

# heart disease webpage function
def heart():
    st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

    st.title("❤️ Heart Disease Prediction App")
    st.write("Enter the patient's details below to check the risk of heart disease.")

    # Collect user input
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", (0, 1), format_func=lambda x: "Female" if x == 0 else "Male")
    cp = st.selectbox("Chest Pain Type (cp)", [0, 1, 2, 3])
    trestbps = st.number_input("Resting Blood Pressure (trestbps)", min_value=50, max_value=250, value=120)
    chol = st.number_input("Serum Cholesterol (chol)", min_value=100, max_value=600, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (fbs)", (0, 1), format_func=lambda x: "False" if x == 0 else "True")
    restecg = st.selectbox("Resting ECG results (restecg)", [0, 1, 2])
    thalach = st.number_input("Max Heart Rate Achieved (thalach)", min_value=50, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina (exang)", (0, 1))
    oldpeak = st.number_input("ST depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope of peak exercise ST segment (slope)", [0, 1, 2])
    ca = st.number_input("Number of major vessels (ca)", min_value=0, max_value=4, value=0)
    thal = st.selectbox("Thalassemia (thal)", [0, 1, 2, 3])

    # Prediction button
    if st.button("Predict"):
        # Prepare data for prediction
        features = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
        
        prediction = heart_model.predict(features)[0]
        
        if prediction == 1:
            st.error("⚠️ High risk of heart disease detected.")
        else:
            st.success("✅ Low risk of heart disease.")

# parkinsons disease webpage function
def parkinsons():

    st.set_page_config(page_title="🧠 Parkinson's Disease Predictor", layout="centered")
    st.title("🧠 Parkinson's Disease Prediction")
    st.markdown("Enter the patient's voice measurement parameters to check the likelihood of Parkinson's disease.")

    # --- Feature Input Section ---
    with st.expander("📊 Fundamental Frequency Features"):
        fo = st.number_input("MDVP:Fo(Hz) – Average vocal fundamental frequency", min_value=50.0, max_value=300.0, value=120.0)
        fhi = st.number_input("MDVP:Fhi(Hz) – Max vocal fundamental frequency", min_value=50.0, max_value=300.0, value=150.0)
        flo = st.number_input("MDVP:Flo(Hz) – Min vocal fundamental frequency", min_value=50.0, max_value=300.0, value=80.0)

    with st.expander("🎯 Jitter and Shimmer Measures"):
        jitter_pct = st.number_input("MDVP:Jitter(%)", min_value=0.0, max_value=0.1, value=0.005)
        jitter_abs = st.number_input("MDVP:Jitter(Abs)", min_value=0.0, max_value=0.001, value=0.00005)
        rap = st.number_input("MDVP:RAP", min_value=0.0, max_value=0.05, value=0.003)
        ppq = st.number_input("MDVP:PPQ", min_value=0.0, max_value=0.05, value=0.005)
        ddp = st.number_input("Jitter:DDP", min_value=0.0, max_value=0.1, value=0.010)
        shimmer = st.number_input("MDVP:Shimmer", min_value=0.0, max_value=1.0, value=0.04)
        shimmer_db = st.number_input("MDVP:Shimmer(dB)", min_value=0.0, max_value=1.0, value=0.4)
        apq3 = st.number_input("Shimmer:APQ3", min_value=0.0, max_value=1.0, value=0.02)
        apq5 = st.number_input("Shimmer:APQ5", min_value=0.0, max_value=1.0, value=0.03)
        apq = st.number_input("MDVP:APQ", min_value=0.0, max_value=1.0, value=0.03)
        dda = st.number_input("Shimmer:DDA", min_value=0.0, max_value=1.0, value=0.06)

    with st.expander("🔬 Other Acoustic Measures"):
        nhr = st.number_input("NHR", min_value=0.0, max_value=1.0, value=0.02)
        hnr = st.number_input("HNR", min_value=0.0, max_value=50.0, value=20.0)
        rpde = st.number_input("RPDE", min_value=0.0, max_value=1.0, value=0.4)
        dfa = st.number_input("DFA", min_value=0.0, max_value=1.0, value=0.8)
        spread1 = st.number_input("Spread1", min_value=-10.0, max_value=0.0, value=-4.5)
        spread2 = st.number_input("Spread2", min_value=0.0, max_value=1.0, value=0.3)
        d2 = st.number_input("D2", min_value=0.0, max_value=5.0, value=2.3)
        ppe = st.number_input("PPE", min_value=0.0, max_value=1.0, value=0.28)

    # --- Prediction Section ---
    if st.button("🔍 Predict"):
        # Prepare feature vector
        features = np.array([[fo, fhi, flo, jitter_pct, jitter_abs, rap, ppq, ddp, shimmer, shimmer_db, apq3, apq5, apq, dda,
                            nhr, hnr, rpde, dfa, spread1, spread2, d2, ppe]])

        # Scale features
        features_scaled = parkinsons_scaler.transform(features)

        # Predict
        pred = parkinsons_model.predict(features_scaled)[0]
        proba = parkinsons_model.predict_proba(features_scaled)[0][1]

        # Show preview
        st.subheader("📄 Input Summary")
        st.dataframe(
            { "Feature": ["Fo", "Fhi", "Flo", "Jitter%", "JitterAbs", "RAP", "PPQ", "DDP", "Shimmer", "Shimmer(dB)", 
                        "APQ3", "APQ5", "APQ", "DDA", "NHR", "HNR", "RPDE", "DFA", "Spread1", "Spread2", "D2", "PPE"],
            "Value": features[0] }
        )

        # Result
        if pred == 1:
            st.error(f"⚠️ Likely Parkinson's Disease detected. (Probability: {proba*100:.2f}%)")
        else:
            st.success(f"✅ Unlikely Parkinson's Disease. (Probability: {proba*100:.2f}%)")


def main():
    # sidebar navigation
    with st.sidebar:
        selected = option_menu(
            "Multiple Disease Diagnosis",
            [   "Breast Cancer",
                "Diabetes",
                "Heart Disease",
                "Parkinsons"
            ],
            icons=[
                "person-standing-dress",
                "capsule",
                "heart-pulse",
                "person-standing"
            ],
            default_index=0
        )

    # navigating selected pages

    # diabetes disease page
    if (selected=="Breast Cancer"):
        breast_cancer()

    if (selected=="Diabetes"):
        diabetes()

    # heart disease page
    if (selected=="Heart Disease"):
        heart()
        
    # parkinsons disease page
    if (selected=="Parkinsons"):
        parkinsons()

if __name__ == "__main__":
    main()