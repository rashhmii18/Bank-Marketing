
import streamlit as st
import pandas as pd
import joblib

model = joblib.load('bank_marketing_rf.pkl')
preprocessor = joblib.load('bank_marketing_preprocessor.pkl')

st.set_page_config(
    page_title="Bank Marketing Predictor",
    page_icon="📊",
    layout="centered"
)

st.title("Bank Marketing Subscription Predictor")
st.write("Enter the client's information to predict whether they are likely to subscribe to a term deposit.")

st.divider()

age = st.number_input("Age", min_value=18, max_value=100, value=35)

job = st.selectbox(
    "Job",
    ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management',
     'retired', 'self-employed', 'services', 'student', 'technician',
     'unemployed', 'unknown']
)

marital = st.selectbox(
    "Marital Status",
    ['married', 'single', 'divorced']
)

education = st.selectbox(
    "Education",
    ['primary', 'secondary', 'tertiary', 'unknown']
)

default = st.selectbox(
    "Credit in Default?",
    ['no', 'yes']
)

balance = st.number_input(
    "Account Balance",
    value=1000
)

housing = st.selectbox(
    "Housing Loan?",
    ['no', 'yes']
)

loan = st.selectbox(
    "Personal Loan?",
    ['no', 'yes']
)

contact = st.selectbox(
    "Contact Type",
    ['cellular', 'telephone', 'unknown']
)

day = st.number_input(
    "Last Contact Day",
    min_value=1,
    max_value=31,
    value=15
)

month = st.selectbox(
    "Last Contact Month",
    ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
     'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
)

campaign = st.number_input(
    "Number of Contacts During This Campaign",
    min_value=1,
    value=1
)

pdays = st.number_input(
    "Days Since Previous Campaign Contact",
    value=-1
)

previous = st.number_input(
    "Number of Previous Contacts",
    min_value=0,
    value=0
)

poutcome = st.selectbox(
    "Previous Campaign Outcome",
    ['unknown', 'failure', 'other', 'success']
)

st.divider()

if st.button("Predict Subscription"):

    input_data = pd.DataFrame({
        'age': [age],
        'job': [job],
        'marital': [marital],
        'education': [education],
        'default': [default],
        'balance': [balance],
        'housing': [housing],
        'loan': [loan],
        'contact': [contact],
        'day': [day],
        'month': [month],
        'campaign': [campaign],
        'pdays': [pdays],
        'previous': [previous],
        'poutcome': [poutcome]
    })

    input_processed = preprocessor.transform(input_data)

    prediction = model.predict(input_processed)[0]
    probability = model.predict_proba(input_processed)[0][1]

    if prediction == 1:
        st.success("The client is predicted to subscribe to a term deposit.")
    else:
        st.info("The client is predicted not to subscribe to a term deposit.")

    st.write(f"Estimated probability of subscription: {probability:.2%}")
