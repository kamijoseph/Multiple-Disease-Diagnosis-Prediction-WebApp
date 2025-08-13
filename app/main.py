
# streamlit web applicatioon
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

# importing the models and scalers
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

diabetes_model = load_pickle("models/diabetes_model.sav")
diabetes_scaler = load_pickle("models/diabetes_scaler.sav")
heart_model = load_pickle("models/heart_model.sav")
parkinsons_model = load_pickle("models/parkinsons_model.sav")
parkinsons_scaler = load_pickle("models/parkinsons_scaler.sav")

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
    st.title("Diabetes Disease Diagnosis.")

# heart disease page
if (selected=="Heart Disease"):
    st.title("Heart Disease Diagnosis.")

# parkinsons disease page
if (selected=="Parkinsons"):
    st.title("Parkinsons Disease Diagnosis.")