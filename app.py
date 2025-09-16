import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Load the model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

# Title of the app
st.title('Neurodegenerative Disease Early Detection')

# Upload dataset
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write(df.head())

    # Preprocess and predict
    X = df[['age', 'memory_score', 'sleep_hours', 'family_history']]
    X_scaled = scaler.transform(X)
    predictions = model.predict(X_scaled)

    # Display predictions
    st.subheader('Predictions:')
    st.write(predictions)