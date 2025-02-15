# Import libraries
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import sklearn
# st.write("Current working directory:", os.getcwd())

# Main title
st.title('Customer Lifetime Value Predictor')
st.text('This web app can be used to predict your CLV (Customer Lifetime Value)')

# Sidebar header for user input
st.sidebar.header("Please input your features")

def create_user_input():
    # Numerical Features
    number_of_policies = st.sidebar.slider('Number of Policies', min_value=1, max_value=9, value=2)
    monthly_premium_auto = st.sidebar.slider('Monthly Premium Auto', min_value=61, max_value=297, value=92)
    total_claim_amount = st.sidebar.slider('Total Claim Amount', min_value=0.423310, max_value=2759.794354, value=429.8)
    income = st.sidebar.slider('Income', min_value=0, max_value=99934, value=37739)
    customer_lifetime_value = st.sidebar.slider('Customer Lifetime Value', min_value=1898.007675, max_value=83325.381190, value=8059.48)

    # Categorical Features
    vehicle_class = st.sidebar.selectbox('Vehicle Class', ['Four-Door Car', 'Two-Door Car', 'SUV', 'Sports Car', 'Luxury SUV', 'Luxury Car'])
    coverage = st.sidebar.selectbox('Coverage', ['Basic', 'Extended', 'Premium'])
    renew_offer_type = st.sidebar.selectbox('Renew Offer Type', ['Offer1', 'Offer2', 'Offer3', 'Offer4'])
    employment_status = st.sidebar.selectbox('Employment Status', ['Employed', 'Unemployed', 'Medical Leave', 'Disabled', 'Retired'])
    marital_status = st.sidebar.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
    education = st.sidebar.selectbox('Education', ['High School or Below', 'College', 'Bachelor', 'Master', 'Doctor'])

    # Creating a dictionary with user input
    user_data = {
        'Number of Policies': number_of_policies,
        'Monthly Premium Auto': monthly_premium_auto,
        'Total Claim Amount': total_claim_amount,
        'Income': income,
        'Customer Lifetime Value': customer_lifetime_value,
        'Vehicle Class': vehicle_class,
        'Coverage': coverage,
        'Renew Offer Type': renew_offer_type,
        'Employment Status': employment_status,
        'Marital Status': marital_status,
        'Education': education
    }
    
    # Convert the dictionary into a pandas DataFrame (for a single row)
    user_data_df = pd.DataFrame([user_data])
    
    return user_data_df

# Get user data from the input
data_customer = create_user_input()

# Create two columns for display
col1, col2 = st.columns(2)

# Left column for displaying input features
with col1:
    st.subheader("Customer's Features")
    st.write(data_customer.transpose())

# Load model
with open(r'clv_model.pkl', 'rb') as f:
    model_loaded = pickle.load(f)
    
# Predict on the user input data
predicted_class = model_loaded.predict(data_customer)
prediction_proba = model_loaded.predict_proba(data_customer)[0]  # Get the probabilities for the classes

# Display the prediction result on the right column
with col2:
    st.subheader('Prediction Result')
    if predicted_class == 1:
        st.write('Class 1: This customer is predicted to have high CLV (valuable customer)')
    else:
        st.write('Class 2: This customer is predicted to have low CLV')

    # Display the probability of high CLV (Class 1)
    st.write(f"Probability of High CLV: {prediction_proba[1]:.2f}")  # Probability for class 1 (high CLV)

