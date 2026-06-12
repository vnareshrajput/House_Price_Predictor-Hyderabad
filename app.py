import streamlit as st
import pandas as pd
import joblib

# ==========================
# Page Configuration
# ==========================
st.set_page_config(
    page_title="Hyderabad House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# ==========================
# Load Model
# ==========================
model = joblib.load("house_price_model.pkl")

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("Hyderabad_houses_cleaned_DS.csv")

# Get unique locations
locations = sorted(df["Location"].dropna().unique())

# ==========================
# Title
# ==========================
st.title("🏠 Hyderabad House Price Predictor")
st.markdown(
    "Enter the property details below to estimate the house price."
)

st.divider()

# ==========================
# Input Layout
# ==========================
col1, col2 = st.columns(2)

with col1:

    area = st.number_input(
        "Area (sq.ft)",
        min_value=100,
        max_value=10000,
        value=1200,
        step=50
    )

    location = st.selectbox(
        "Location",
        locations
    )

    bedrooms = st.number_input(
        "No. of Bedrooms",
        min_value=1,
        max_value=10,
        value=2
    )

    resale = st.selectbox(
        "Resale Property",
        ["No", "Yes"]
    )
    resale = 1 if resale == "Yes" else 0


with col2:

    parking = st.selectbox(
        "Car Parking",
        ["No", "Yes"]
    )
    parking = 1 if parking == "Yes" else 0

    swimming_pool = st.selectbox(
        "Swimming Pool",
        ["No", "Yes"]
    )
    swimming_pool = 1 if swimming_pool == "Yes" else 0

    jogging_track = st.selectbox(
        "Jogging Track",
        ["No", "Yes"]
    )
    jogging_track = 1 if jogging_track == "Yes" else 0

    lift = st.selectbox(
        "Lift Available",
        ["No", "Yes"]
    )
    lift = 1 if lift == "Yes" else 0


# ==========================
# Prediction Button
# ==========================
if st.button("🔮 Predict House Price"):

    input_df = pd.DataFrame({
        "Area": [area],
        "Location": [location],
        "No. of Bedrooms": [bedrooms],
        "Resale": [resale],
        "CarParking": [parking],
        "SwimmingPool": [swimming_pool],
        "JoggingTrack": [jogging_track],
        "LiftAvailable": [lift]
    })

    prediction = model.predict(input_df)[0]

    st.success(
        f"🏠 Estimated House Price: ₹ {prediction:,.2f}"
    )

    st.subheader("Property Summary")

    st.dataframe(input_df)