import streamlit as st
import pickle
import pandas as pdimport streamlit as st
import pickle
import pandas as pd
import os

# Load the model and encoder
@st.cache_resource
def load_model():
    model_path = os.path.join("notebooks", "best_lightgbm_model.pkl")
    ohe_path = os.path.join("notebooks", "ohe.pkl")
    model = pickle.load(open(model_path, "rb"))
    ohe = pickle.load(open(ohe_path, "rb"))
    return model, ohe

# ❗ Call the function here to get model and encoder
model, ohe = load_model()

# Streamlit UI
st.title("Push Notification Optimizer")

st.markdown("### Enter notification features to predict the open rate")

col1, col2 = st.columns(2)

with col1:
    category = st.selectbox("Category", ['General', 'Discount', 'Urgent', 'New Launch'])
    urgency = st.selectbox("Urgency Level", ['Low', 'Medium', 'High'])

with col2:
    user_segment = st.selectbox("User Segment", ['New', 'Active', 'Inactive'])
    day = st.selectbox("Day of the Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

if st.button("Predict Open Rate"):
    input_df = pd.DataFrame([[category, urgency, user_segment, day]],
                            columns=["category", "urgency", "user_segment", "day"])

    # Apply one-hot encoding
    encoded_input = ohe.transform(input_df).toarray()

    # Predict
    prediction = model.predict(encoded_input)

    st.success(f"📬 Predicted Open Rate: {prediction[0]*100:.2f}%")
