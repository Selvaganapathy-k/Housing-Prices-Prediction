import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained model and columns
with open('xgb_house_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('house_model_columns.pkl', 'rb') as f:
    columns = pickle.load(f)

st.title("California Median House Value Predictor")

st.write("Fill all the fields below to predict the median house price:")

def user_input_features():
    longitude = st.number_input('Longitude', value=-120.0)
    latitude = st.number_input('Latitude', value=35.0)
    housing_median_age = st.number_input('Housing Median Age', min_value=1, max_value=100, value=20)
    total_rooms = st.number_input('Total Rooms', min_value=1, max_value=int(1e5), value=1000)
    total_bedrooms = st.number_input('Total Bedrooms', min_value=1, max_value=int(1e4), value=200)
    population = st.number_input('Population', min_value=1, max_value=int(1e5), value=1000)
    households = st.number_input('Households', min_value=1, max_value=int(1e4), value=300)
    median_income = st.number_input('Median Income (in tens of thousands)', min_value=0.1, max_value=15.0, value=4.0)
    
    # One-hot encoding for 'ocean_proximity'
    op = st.selectbox('Ocean Proximity', ['INLAND', '<1H OCEAN', 'NEAR BAY', 'NEAR OCEAN', 'ISLAND'])
    op_dict = {f'ocean_proximity_{cat}': 0 for cat in ['INLAND', '<1H OCEAN', 'NEAR BAY', 'NEAR OCEAN', 'ISLAND']}
    op_dict[f'ocean_proximity_{op}'] = 1

    feature_dict = {
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income,
    }
    feature_dict.update(op_dict)
    
    # Make sure all columns are in the correct order
    return pd.DataFrame([feature_dict])[columns]

input_df = user_input_features()

if st.button('Predict'):
    prediction = model.predict(input_df)[0]
    st.success(f"Predicted median house value: ${prediction:,.0f}")

st.write("Model trained on California housing data.")
