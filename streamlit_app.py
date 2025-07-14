import streamlit as st
import numpy as np
import pickle

# Load model and scaler
try:
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    st.warning("⚠️ Running in Demo Mode (Model not loaded)")

# Streamlit app UI
st.set_page_config(page_title="Smart Maintenance Priority", page_icon="🔧", layout="centered")

st.title("🔧 Smart Maintenance Priority Prediction")
st.markdown("Enter the machine inspection data to predict maintenance priority.")

# Input fields
temp = st.text_input("🌡️ Temperature (°C)", "e.g., 75")
vibration = st.text_input("🔊 Vibration (mm/s)", "e.g., 1.5")
pressure = st.text_input("⚙️ Pressure (bar)", "e.g., 5.0")
inspection = st.text_input("🕒 Inspection Duration (min)", "e.g., 30")
downtime = st.text_input("💰 Downtime Cost (USD)", "e.g., 1000")
technician = st.text_input("👷 Technician Availability (%)", "e.g., 85")

# Predict button
if st.button("🔮 Predict Maintenance Priority"):
    try:
        inputs = [temp, vibration, pressure, inspection, downtime, technician]
        if any(i.strip().startswith("e.g.") or i.strip() == "" for i in inputs):
            st.error("❗ Please fill in all the input fields correctly.")
        else:
            input_data = np.array([[float(temp), float(vibration), float(pressure),
                                    float(inspection), float(downtime), float(technician)]])
            
            if MODEL_LOADED:
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)[0]
            else:
                prediction = np.random.choice(['Low', 'Medium', 'High'])

            st.success(f"🎯 Predicted Maintenance Priority: **{prediction}**")

            # Show input summary
            st.subheader("📋 Input Summary")
            st.markdown(f"""
            - Temperature: {temp} °C  
            - Vibration: {vibration} mm/s  
            - Pressure: {pressure} bar  
            - Inspection Duration: {inspection} min  
            - Downtime Cost: ${downtime}  
            - Technician Availability: {technician}%
            """)
    except Exception as e:
        st.error(f"An error occurred: {e}")
