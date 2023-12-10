#import all required packages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu


header=st.beta_container()
dataset=st.beta_container()
features=st.beta_container()
model_training=st.beta_container()

with header:
    st.title ("Welcome to Genesis' LinkedIn user predictor")
    st.text ("In this website, you will be provide your input and we will predict if you use LinkedIn or not")

with dataset:
    ss=pd.read_csv("part2_ss_appfile.csv")
    ss = ss.dropna()

with features: 
    st.header ('The following are information that you enter into the predictor')

    st.markdown

    
    