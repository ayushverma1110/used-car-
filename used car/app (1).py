import streamlit as st
import requests
import pandas as pd

# --------------------------------------------------
# Page config
# --------------------------------------------------
st.set_page_config(page_title="Used Car Price Predictor")

st.title("🚗 Used Car Price Prediction")

# --------------------------------------------------
# Load dataset (ABSOLUTE PATH - WINDOWS SAFE)
# --------------------------------------------------
DATA_PATH = r"E:\Projects\Used Car Price\data\used_cars.csv"

df = pd.read_csv(DATA_PATH)

# Drop unwanted column if present
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# --------------------------------------------------
# Input fields
# --------------------------------------------------
year = st.number_input("Manufacturing Year", 2000, 2025, 2022)
km = st.number_input("Kilometers Driven", min_value=0)

# -------- Brand dropdown --------
brands = sorted(df['brand'].dropna().astype(str).unique())
brand = st.selectbox("Brand", brands)

# -------- Model dropdown (depends on brand) --------
models = sorted(
    df[df['brand'] == brand]['model']
    .dropna()
    .astype(str)
    .unique()
)
model = st.selectbox("Model", models)

# -------- Other dropdowns (type-safe) --------
fuel = st.selectbox(
    "Fuel Type",
    sorted(df['fuel_type'].dropna().astype(str).unique())
)

trans = st.selectbox(
    "Transmission",
    sorted(df['transmission_type'].dropna().astype(str).unique())
)

city = st.selectbox(
    "City",
    sorted(df['city'].dropna().astype(str).unique())
)

body = st.selectbox(
    "Body Type",
    sorted(df['bodytype'].dropna().astype(str).unique())
)

owners = st.selectbox(
    "Number of Owners",
    sorted(df['number_of_owners'].dropna().astype(int).unique())
)

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if st.button("Predict Price"):
    payload = {
        "manufacturing_year": int(year),
        "km_driven": int(km),
        "brand": brand,
        "model": model,
        "fuel_type": fuel,
        "transmission_type": trans,
        "city": city,
        "bodytype": body,
        "number_of_owners": int(owners)
    }

    try:
        response = requests.post(
        "http://127.0.0.1:8000/predict",
        json=payload,
        timeout=5
    )

        if response.status_code == 200:
            price = response.json()["predicted_price"]
            st.success(f"💰 Estimated Price: ₹ {price}")
        else:
            st.error(response.text)

    except Exception as e:
        st.error(f"❌ Error: {e}")


