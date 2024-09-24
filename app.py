import streamlit as st
import pandas as pd
from functions import *


st.title('ML on Web')

model_sidebar = st.sidebar.radio("Choose a ML Model", ('Linear Regression', 'Logistic Regression','EDA Graphs'))
file_upload = st.file_uploader("Upload Dataset",type='csv')


try:
    df = pd.read_csv(file_upload)
    col1 , col2 = st.columns(2)
    with col1:
        vars = st.multiselect("Choose the Variables",df.columns)
        X = df[vars]
        st.write(X)
    with col2:
        target = st.selectbox("Choose the Target Variable",df.columns)
        Y = df[target]
        st.write(Y)
    adv_setting = st.toggle("Advanced Settings")
    if adv_setting:
        test_train_ratio = st.slider("Test Train Split Ratio",0.0,1.0,0.2)
        random_states = st.text_input("Enter Random State")

    if Y.name in X.columns:
        st.write("Target variable should not be in the list of variables")
    else:
        submitted = st.button("Cook it")
    if submitted:
        if model_sidebar == 'Linear Regression':
            Linearregression(X,Y)
        if model_sidebar =='Logistic Regression':
            Logisticregression(X,Y)
        
except ValueError:
    st.write("Please upload a dataset")




