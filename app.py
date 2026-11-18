
import streamlit as st
import pandas as pd
import joblib

FEATURES = ['Pclass', 'Sex', 'Age', 'Fare']

@st.cache_resource
def load_model():
    return joblib.load('model.joblib')

st.title("Titanic Survival Prediction App")

st.write("Enter passenger details to predict the estimated survival probability.")

pclass = st.selectbox("Ticket Class", [1, 2, 3])

sex = st.radio("Sex", ["Female", "Male"])

age = st.slider("Age", 1, 80, 30)

fare = st.slider("Fare Paid", 0, 500, 30)

if st.button("Predict"):

    try:
        model = load_model()

        sex_value = 1 if sex == "Female" else 0

        data = pd.DataFrame(
            [[pclass, sex_value, age, fare]],
            columns=FEATURES
        )

        prediction = model.predict(data)

        probability = model.predict_proba(data)[0][1]

        st.metric(
            "Estimated Survival Probability",
            f"{probability:.0%}"
        )

        if prediction[0] == 1:
            st.success("The model predicts that the passenger may have survived.")
        else:
            st.warning("The model predicts that the passenger may not have survived.")

    except FileNotFoundError:
        st.error("Model file is missing. Please upload model.joblib.")

st.caption("Educational machine learning project. Predictions are not guaranteed.")
