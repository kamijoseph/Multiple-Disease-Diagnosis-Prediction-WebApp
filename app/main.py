
# streamlit web applicatioon
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import numpy as np

# importing the models and scalers
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

diabetes_model = load_pickle("models/diabetes_model.sav")
diabetes_scaler = load_pickle("models/diabetes_scaler.sav")
heart_model = load_pickle("models/heart_model.sav")
parkinsons_model = load_pickle("models/parkinsons_model.sav")
parkinsons_scaler = load_pickle("models/parkinsons_scaler.sav")

# diabetes predictive system
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

# diabetes disease webpage function
def diabetes():
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
            [
                "Diabetes",
                "Heart Disease",
                "Parkinsons"
            ],
            icons=[
                "capsule",
                "heart-pulse",
                "person-standing"
            ],
            default_index=0
        )

    # navigating selected pages

    # diabetes disease page
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