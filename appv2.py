#import all required packages
import pandas as pd
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
from sklearn.linear_model import LogisticRegression


header=st.container()
features=st.container()

with header:
    st.title ("Welcome to Genesis' LinkedIn user predictor")
    st.text ("In this website, you will be provide your input and we will predict if you use LinkedIn or not")

    ss=pd.read_csv("part2_ss_appfile.csv")
    ss = ss.dropna()
    ss.info


with features: 
    st.header ('The following are information that you enter into the predictor')

    st.markdown ("- Income    (1-9) - Between $0 to +$150k")  
    st.markdown("- Education (1-8) - Highest level of school/ degree completed")   
    st.markdown( "- Parent          - Yes or No")  
    st.markdown("- Marital  (0-5)  - Current marital status")  
    st.markdown("- Gender          - Male or Female")   
    st.markdown("- Age             - Current age")

    
#from dataframe ss: y as sm_li; x as income, education, parent, marital, gender, age
y = ss["sm_li"]
X = ss[["income", "education", "parent", "marital","gender","age"]]
lr= LogisticRegression(class_weight="balanced").fit(X,y)
lr.fit(X,y)
y_pred = lr.predict(X)

 
with st.sidebar:
    income = st.number_input("Income (low=1 to high=9)", 1, 9)
    education=st.number_input('What is the highest level of education',1,8)
    marital=st.number_input('What is your current marital status',1,6)
    parent=st.slider('Do you have children',0,1)
    gender=st.slider('Gender',0,1)
    age=st.slider('What is age',0,98)


# Create labels from numeric inputs
 # Income
if income == 1:
    inc_label = "Less than $10,000"
elif income == 2:
    inc_label = "10 to under $20,000"
elif income == 3:
    inc_label = "20 to under $30,000"
elif income == 4:
    inc_label = "30 to under $40,000"    
elif income == 5:
    inc_label = "40 to under $50,000"  
elif income == 6:
    inc_label = "50 to under $75,000"
elif income == 7:
    inc_label = "75 to under $100,000"  
elif income == 8:
    inc_label = "100 to under $150,000"      
else:
    inc_label = "$150,000 +" 

# Education
education = st.selectbox("Education level", 
             options = ["Less than high school",
                       "Some high high school",
                        "High school graduate/ GED",
                        "Some college",
                        "Two-year college/ Associates",
                        "Four-year college/ Bachelors",
                        "Some post-graduate or professional schooling",
                        "Graduate or professional degree (MA, MS, Phd, MD, JD, others"])

st.write(f"Education (pre-conversion): {education}")

if education == "Less than high school":
    education = 1
elif education == "Some high high school":
    education = 2
elif education == "High school graduate/ GED":
    education = 3
elif education == "Some college":
    education = 4
elif education == "Two-year college/ Associates":
    education = 5
elif education == "Four-year college/ Bachelors":
    education = 6
elif education == "Some post-graduate or professional schooling":
    education = 7
else:
    education = 8


# Marital
if marital == 1:
    mar_label = "Married"
elif marital == 2:
    mar_label = "Living with a partner"
elif marital == 3:
    mar_label = "Divorced"
elif marital == 4:
    mar_label = "Separated"
elif marital == 5:
    mar_label = "Never been married/ Single"
else:
    mar_label = "Widowed"

#Parent
parent = st.slider(label="Are you a parent of a child under 18 living in your home? (0 Yes, 1 No)", 
           min_value=0,
           max_value=1,
           value=1)

#Gender
gender = st.slider(label="Gender (0 male, 1 female)", 
           min_value=0,
           max_value=1,
           value=1)

#Age
Age = st.slider(label="Age", 
           min_value=18,
           max_value=98,
           value=45)



#Making predictions 
##New data for features: age, college, high_income, ideology
person = [{income}, {education}, {parent}, {marital},{gender},{age}]
##Predict class, given input features
predicted_class = lr.predict([person])
##Generate probability of positive class (=1)
probs = lr.predict_proba([person])


##Print predicted class and probability
st.write(f"Predicted class: {predicted_class[0]}") # 0=Doesn't use LinkedIn, 1=Uses LinkedIN
st.write(f"Probability that this person is LinkedIn user: {probs[0][1]}")
