import streamlit as st
import numpy as np
import pickle
from keras.models import load_model
from warnings import filterwarnings
filterwarnings("ignore")

# ======================================
# Load ANN Model and Scaler
# ======================================

model = load_model("housing_price_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))

# ======================================
# Page Configuration
# ======================================

st.set_page_config(
    page_title="Boston Housing Prediction",
    page_icon="🏠",
    layout="centered"
)

# ======================================
# Title and Description
# ======================================

st.title("🏠 Boston Housing Price Prediction App")

st.markdown("""
This application predicts Boston house prices using an Artificial Neural Network (ANN) model.
""")

st.sidebar.header("Enter Housing Features")

# ======================================
# Input Fields
# ======================================

col1, col2 = st.columns(2)

with col1:
    crim = st.number_input("CRIM - Crime Rate", min_value=0.0)
    indus = st.number_input("INDUS - Industrial Area", min_value=0.0)
    nox = st.number_input("NOX - Nitric Oxide", min_value=0.0)
    age = st.number_input("AGE - Old Houses Percentage", min_value=0.0)
    rad = st.number_input("RAD - Highway Accessibility", min_value=0)
    ptratio = st.number_input("PTRATIO - Pupil Teacher Ratio", min_value=0.0)
    lstat = st.number_input("LSTAT - Lower Status Population %", min_value=0.0)

with col2:
    zn = st.number_input("ZN - Residential Land", min_value=0.0)
    chas = st.selectbox("CHAS - Charles River", [0, 1])
    rm = st.number_input("RM - Average Rooms", min_value=0.0)
    dis = st.number_input("DIS - Distance to Employment Centers", min_value=0.0)
    tax = st.number_input("TAX - Property Tax Rate", min_value=0.0)
    b = st.number_input("B - Black Population Index", min_value=0.0)

# ======================================
# Prediction Button
# ======================================

if st.button("Predict House Price"):

    # Feature array
    features = np.array([[
        crim,
        zn,
        indus,
        chas,
        nox,
        rm,
        age,
        dis,
        rad,
        tax,
        ptratio,
        b,
        lstat
    ]])

    # Scale features
    features_scaled = scaler.transform(features)

    # Prediction
    prediction = model.predict(features_scaled)

    # Display result
    st.success(f"🏡 Predicted House Price: ${prediction[0][0]:.2f}")

# ======================================
# Footer
# ======================================

st.markdown("---")
st.markdown("Built with Streamlit and TensorFlow")