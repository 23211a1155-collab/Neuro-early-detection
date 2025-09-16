
import streamlit as st
import numpy as np
import pickle

# Load the model and scaler files
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

st.title("Neurodegenerative Disease Early Detection")

st.write("""
This app predicts the risk of early-stage neurodegenerative disease based on user inputs.
""")

# Input fields
age = st.slider("Age", 40, 90, 65)
memory_score = st.slider("Memory Test Score (0-30)", 0, 30, 15)
sleep_hours = st.slider("Average Sleep Hours", 3, 10, 6)
family_history = st.radio("Family History of Disease", ("No", "Yes"))
family_history = 1 if family_history == "Yes" else 0

# When user clicks Predict
if st.button("Predict"):
    # Prepare the input
    input_data = np.array([[age, memory_score, sleep_hours, family_history]])
    input_scaled = scaler.transform(input_data)
    
    # Predict
    prediction = model.predict(input_scaled)[0]
    diagnosis = ["Healthy", "At Risk", "Early Stage"]
    result = diagnosis[prediction]
    
    # Show result
    st.subheader("Prediction Result")
    st.write(f"The predicted condition is: **{result}**")
