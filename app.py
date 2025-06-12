import streamlit as st
import pandas as pd
import joblib

@st.cache_resource
def load_model():
    model = joblib.load("best_lightgbm_model.pkl")
    ohe = joblib.load("ohe.pkl")
    return model, ohe

model, ohe = load_model()

st.title("📬 Push Notification Optimizer")
st.markdown("### Enter notification features to predict the open rate")

col1, col2 = st.columns(2)

with col1:
    category = st.selectbox("Category", ['General', 'Discount', 'Urgent', 'New Launch'])
    urgency = st.selectbox("Urgency Level", ['Low', 'Medium', 'High'])

with col2:
    user_segment = st.selectbox("User Segment", ['New', 'Active', 'Inactive'])
    day = st.selectbox("Day of the Week", ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

if st.button("Predict Open Rate"):
    input_df = pd.DataFrame({
        "category": [category],
        "urgency": [urgency],
        "user_segment": [user_segment],
        "day": [day]
    })

    try:
        # Safely transform the input
        encoded_input = ohe.transform(input_df)

        # Convert to array if needed
        if hasattr(encoded_input, "toarray"):
            encoded_input = encoded_input.toarray()

        prediction = model.predict(encoded_input)
        st.success(f"📈 Predicted Open Rate: {prediction[0]*100:.2f}%")

    except Exception as e:
        st.error("❌ Prediction Failed")
        st.code(str(e))
        st.write("Debug input:")
        st.dataframe(input_df)




