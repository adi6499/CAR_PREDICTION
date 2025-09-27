import os        # Make sure this is imported at the top
import pickle
import pandas as pd
import streamlit as st  # Only needed if using Streamlit

# Get the directory where this script is located
with open("pipeline.pkl", "rb") as f:
    model = pickle.load(f)

car_data = pd.read_csv("cleaned_car.csv")
# Example for Streamlit input & prediction
st.title("Car Price Prediction")

name = st.selectbox("Select Car Model", car_data["name"].unique())
company = st.selectbox("Select Car Company", car_data["company"].unique())
fuel_type = st.selectbox("Select Fuel Type", car_data["fuel_type"].unique())
year = st.number_input("Enter Year of Manufacture", min_value=1990, max_value=2025, step=1)
kms_driven = st.number_input("Enter Kilometers Driven", min_value=0, max_value=500000, step=1000)


if st.button("Predict Price"):
    input_df = pd.DataFrame([{
        "year": year,
        "kms_driven": kms_driven,
        "company": company,
        "fuel_type": fuel_type,
        "name": name
    }])
    
    prediction = model.predict(input_df)
    st.success(f"Predicted Price: ₹{int(prediction[0])}")
