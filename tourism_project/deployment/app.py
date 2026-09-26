"""
Streamlit web app for the "Visit with Us" Wellness Tourism Package
purchase-prediction model.

The app first tries to download the registered model from the Hugging
Face Hub Model repo (set MODEL_REPO_ID / HF_TOKEN as needed). If that is
not available (e.g. no internet access, no token), it falls back to the
model file that was saved locally by train.py, so the app also works for
local/offline testing.
"""
import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_REPO_ID = os.getenv("MODEL_REPO_ID", "<your-hf-username>/tourism-wellness-package-model")
LOCAL_MODEL_PATH = os.path.join(os.path.dirname(__file__), "best_tourism_model_v1.joblib")


@st.cache_resource
def load_model():
    # 1. Try to pull the model registered on the Hugging Face Hub
    try:
        from huggingface_hub import hf_hub_download
        model_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename="best_tourism_model_v1.joblib",
            token=os.getenv("HF_TOKEN"),
        )
        return joblib.load(model_path)
    except Exception:
        pass

    # 2. Fall back to a local copy shipped alongside the app
    if os.path.exists(LOCAL_MODEL_PATH):
        return joblib.load(LOCAL_MODEL_PATH)

    return None


st.set_page_config(page_title="Wellness Tourism Package Predictor", page_icon="\U0001F9F3")
st.title("\U0001F9F3 Wellness Tourism Package - Purchase Predictor")
st.write(
    "Predict whether a customer is likely to purchase the new **Wellness "
    "Tourism Package**, based on their profile and their interaction with "
    "the sales pitch."
)

model = load_model()
if model is None:
    st.error(
        "No trained model could be found. Please run `train.py` first, or "
        "set `MODEL_REPO_ID` to a valid Hugging Face model repository."
    )
    st.stop()

st.header("Customer Profile")
col1, col2, col3 = st.columns(3)

with col1:
    Age = st.number_input("Age", min_value=18, max_value=100, value=35)
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    Occupation = st.selectbox(
        "Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"]
    )
    Gender = st.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])

with col2:
    NumberOfPersonVisiting = st.number_input("Number of Persons Visiting", 1, 10, 3)
    NumberOfChildrenVisiting = st.number_input("Number of Children Visiting (< age 5)", 0, 5, 0)
    NumberOfTrips = st.number_input("Avg Number of Trips per Year", 0, 20, 2)
    PreferredPropertyStar = st.selectbox("Preferred Property Star Rating", [3.0, 4.0, 5.0])
    Passport = st.selectbox("Holds Valid Passport?", ["Yes", "No"])
    OwnCar = st.selectbox("Owns a Car?", ["Yes", "No"])

with col3:
    Designation = st.selectbox(
        "Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"]
    )
    MonthlyIncome = st.number_input("Monthly Income", 1000, 100000, 22000, step=500)
    ProductPitched = st.selectbox(
        "Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"]
    )
    NumberOfFollowups = st.number_input("Number of Follow-ups", 0, 10, 3)
    DurationOfPitch = st.number_input("Duration of Pitch (minutes)", 0, 60, 10)
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)

if st.button("Predict Purchase Likelihood", type="primary"):
    input_df = pd.DataFrame([{
        "Age": Age,
        "TypeofContact": TypeofContact,
        "CityTier": CityTier,
        "DurationOfPitch": DurationOfPitch,
        "Occupation": Occupation,
        "Gender": Gender,
        "NumberOfPersonVisiting": NumberOfPersonVisiting,
        "NumberOfFollowups": NumberOfFollowups,
        "ProductPitched": ProductPitched,
        "PreferredPropertyStar": PreferredPropertyStar,
        "MaritalStatus": MaritalStatus,
        "NumberOfTrips": NumberOfTrips,
        "Passport": 1 if Passport == "Yes" else 0,
        "PitchSatisfactionScore": PitchSatisfactionScore,
        "OwnCar": 1 if OwnCar == "Yes" else 0,
        "NumberOfChildrenVisiting": NumberOfChildrenVisiting,
        "Designation": Designation,
        "MonthlyIncome": MonthlyIncome,
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")
    if prediction == 1:
        st.success(f"\u2705 Likely to purchase the Wellness Package (probability: {probability:.1%})")
    else:
        st.warning(f"\u274C Unlikely to purchase the Wellness Package (probability: {probability:.1%})")
