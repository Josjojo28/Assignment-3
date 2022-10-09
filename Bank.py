import streamlit as st
import joblib
import numpy as np
st.set_page_config(layout="wide", initial_sidebar_state="expanded", page_icon="⚡",
                   page_title="Bank Churn Model")
model = "./Assignment 3/banking_churn.csv"
st.title("Bank Churn Model")
html_temp = """
<div style ="background-color:#025246 ;padding:10px">
<h2 style="color:white;text-align:center;">Bank Churn ML App</h2>
</div>
"""
st.markdown(html_temp, unsafe_allow_html=True)
Gender = st.text_input("Gender")
Age = st.text_input("Geography")
Age = st.text_input("Age")
Tenure = st.text_input("Tenure", )
Balance = st.text_input("Balance")
HasCrCard = st.text_input("HasCrCard")
EstimatedSalary = st.text_input("EstimatedSalary")

safe_html = """
<div style ="background-color:#F4D03F;padding:10px>
<h2 style="color:white; text-align:center;">Customer will not exit</h2>
</div>
"""
danger_html = """
<div style ="background-color:#F4D03F;padding:10px>
<h2 style="color:white; text-align:center;">Customer will exit</h2>
</div>
"""


def predict_cust(Age, Tenure, Balance, HasCrCard, EstimatedSalary):
    input = np.array([[Age, Tenure, Balance, HasCrCard, EstimatedSalary]]).astype(np.float64)
    prediction = model.predict_proba(input)
    pred = np.argmax(prediction)
    return pred


if st.button("Predict"):
    output = predict_cust(Age, Tenure, Balance, HasCrCard, EstimatedSalary)
    st.success("The verdict{}".format(output))

    if output == 0:
        st.markdown(safe_html, unsafe_allow_html=True)
    else:
        st.markdown(danger_html, unsafe_allow_html=True)



