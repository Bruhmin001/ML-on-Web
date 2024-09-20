import streamlit as st
from functions import * 
st.title('ML on Web')

model_sidebar = st.sidebar.radio("Choose a ML Model", ('Linear Regression', 'Logistic Regression'))
start_button = st.button("Start")
if start_button:
    start()
