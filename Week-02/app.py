import streamlit as st
import pandas as pd
import pickle
import numpy as np

import os

# Set page configuration
st.set_page_config(page_title="Telecom Customer Churn Predictor", layout="wide")

# App Header
st.title("📞 Telecom Customer Churn Prediction Dashboard")
st.markdown("""
Predict if a customer will leave the service based on their profile and usage.
""")

# Load the model and columns
@st.cache_resource
def load_assets():
    # Get the directory of the current script
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, 'churn_model.pkl')
    cols_path = os.path.join(base_path, 'columns.pkl')
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(cols_path, 'rb') as f:
            cols = pickle.load(f)
        return model, cols
    except FileNotFoundError:
        return None, None

model, model_columns = load_assets()

if model is None:
    st.error("Model files not found. Please run the Jupyter Notebook first to train and save the model.")
else:
    # Sidebar for user inputs
    st.sidebar.header("Customer Profile")
    
    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    tenure = st.sidebar.slider("Tenure (Months)", 0, 72, 12)
    
    st.sidebar.header("Service Details")
    phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
    backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    protection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    support = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    
    st.sidebar.header("Contract & Billing")
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly = st.sidebar.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
    total = monthly * tenure # Approximate total charges

    # Data transformation
    def preprocess_input():
        # Map inputs back to model format
        # We need to create a tenure group as well
        def get_tenure_group(t):
            if t <= 12: return '0-1 Year'
            elif t <= 24: return '1-2 Years'
            elif t <= 48: return '2-4 Years'
            elif t <= 60: return '4-5 Years'
            else: return '> 5 Years'
        
        data = {
            'gender': 1 if gender == "Male" else 0,
            'SeniorCitizen': senior,
            'Partner': 1 if partner == "Yes" else 0,
            'Dependents': 1 if dependents == "Yes" else 0,
            'PhoneService': 1 if phone == "Yes" else 0,
            'PaperlessBilling': 1 if billing == "Yes" else 0,
            'MonthlyCharges': monthly,
            'TotalCharges': total,
            'LTV': monthly * tenure,
            'MultipleLines_' + multiple: 1,
            'InternetService_' + internet: 1,
            'OnlineSecurity_' + security: 1,
            'OnlineBackup_' + backup: 1,
            'DeviceProtection_' + protection: 1,
            'TechSupport_' + support: 1,
            'StreamingTV_' + tv: 1,
            'StreamingMovies_' + movies: 1,
            'Contract_' + contract: 1,
            'PaymentMethod_' + payment: 1,
            'TenureGroup_' + get_tenure_group(tenure): 1
        }
        
        # Create a dataframe with all zeros for model columns
        input_df = pd.DataFrame(columns=model_columns)
        input_df.loc[0] = 0
        
        # Fill in the data we have
        for key, value in data.items():
            if key in input_df.columns:
                input_df.at[0, key] = value
        
        return input_df

    # Main area
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Summary")
        st.write(f"**Tenure:** {tenure} months")
        st.write(f"**Monthly Charges:** ${monthly}")
        st.write(f"**Estimated Total:** ${total}")
    
    with col2:
        st.subheader("Prediction Result")
        processed_data = preprocess_input()
        prediction_prob = model.predict_proba(processed_data)[0][1]
        prediction = model.predict(processed_data)[0]
        
        if prediction == 1:
            st.error(f"High Risk of Churn! ({prediction_prob:.2%})")
        else:
            st.success(f"Low Risk of Churn ({prediction_prob:.2%})")

    st.progress(prediction_prob)

    st.divider()
    st.info("💡 Tip: Try changing the 'Contract' type to see how it affects the churn probability.")
