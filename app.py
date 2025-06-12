import streamlit as st
import pandas as pd
import pickle

# Load your model and encoder
@st.cache_resource
def load_model():
    model = pickle.load(open("best_lightgbm_model.pkl", "rb"))
    ohe = pickle.load(open("ohe.pkl", "rb"))
    return model, ohe

model, ohe = load_model()

# UI
st.title("📲 Push Notification Optimizer")
st.write("Fill in the details below to get the best time and audience for push notifications!")

# Example Inputs (Replace with your real features)
platform = st.selectbox("Platform", ["Android", "iOS"])
segment = st.selectbox("User Segment", ["New", "Returning"])
hour = st.slider("Hour of Day", 0, 23)

# Make prediction
if st.button("Predict Engagement Score"):
    input_df = pd.DataFrame([[platform, segment, hour]], columns=["platform", "segment", "hour"])
    input_encoded = ohe.transform(input_df)
    prediction = model.predict(input_encoded)
    st.success(f"📈 Predicted Engagement Score: {prediction[0]:.2f}")

