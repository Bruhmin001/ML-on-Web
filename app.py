import streamlit as st
import pandas as pd

st.title('ML on Web')

model_sidebar = st.sidebar.radio("Choose a ML Model", ('Linear Regression', 'Logistic Regression'))
file_upload = st.file_uploader("Upload Dataset",type='csv')
try:
    df = pd.read_csv(file_upload)
    df.to_csv("from_user.csv")
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
        test_train_ratio = st.slider("Test Train Split Ratio",0.0,1.0,0.8)
        random_state = st.text_input("Enter Random State")
    if Y.name in X.columns:
        st.write("Target variable should not be in the list of variables")
    else:
        st.button("Cook it")

except ValueError:
    st.write("Please upload a file") 

