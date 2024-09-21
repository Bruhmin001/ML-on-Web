import streamlit as st
import pandas as pd
from app import *

df = pd.read_csv("from_user.csv")
print(df)

def get_column(df):
    return df.columns

X,Y = send_data()