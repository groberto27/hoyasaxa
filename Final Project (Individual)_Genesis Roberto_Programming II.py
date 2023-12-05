#!/usr/bin/env python
# coding: utf-8

# # Final Project

# ## Genesis Roberto

# ## Dec 12, 2023

# ***

# ### Part 1 (80%): Building a classification model to predict LinkedIn users

# #### 1. Read in the data, call the dataframe "s"  and check the dimensions of the dataframe

# In[1]:


#import all required packages
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import altair as alt


s=pd.read_csv("social_media_usage.csv")
s.info
s.shape


# In[2]:


pd.options.display.max_columns = None
s.head()


# #### 2. Define a function called clean_sm that takes one input, x, and uses `np.where` to check whether x is equal to 1. If it is, make the value of x = 1, otherwise make it 0. Return x. Create a toy dataframe with three rows and two columns and test your function to make sure it works as expected

# In[3]:


def clean_sm(x):
    clean_sm=np.where(x==1,1,0)
    return(clean_sm)


# In[4]:


toy_df=pd.DataFrame({'A':[1,2,9],
                     'B':[-1,0,1]})
toy_df


# In[5]:


clean_sm(toy_df)


# #### 3. Create a new dataframe called "ss". The new dataframe should contain a target column called sm_li which should be a binary variable ( that takes the value of 1 if it is 1 and 0 otherwise (use clean_sm to create this) which indicates whether or not the individual uses LinkedIn, and the following features: income (ordered numeric from 1 to 9, above 9 considered missing), education (ordered numeric from 1 to 8, above 8 considered missing), parent (binary), married (binary), female (binary), and age (numeric, above 98 considered missing). Drop any missing values. Perform exploratory analysis to examine how the features are related to the target.

# In[6]:


#create new clean data frame with sm_li as target value
ss=pd.DataFrame({
    "sm_li":clean_sm(s["web1h"]),
    "income":np.where(s["income"] <=9, s["income"], np.nan),
    "education":np.where(s["educ2"] <=8,s["educ2"], np.nan),
    "parent":np.where(s["par"] == 1, 1,0),
    "marital":np.where(s["marital"] ==1, 1,0),
    "gender":np.where(s["gender"] ==1, 1,0),
    "age":np.where(s["age"] <=98, s["age"], np.nan)})


# In[7]:


ss.head()


# In[8]:


ss.shape #before dropping missing values


# In[9]:


type(ss)


# In[10]:


# Identify missing data
ss.isnull().sum()  #number of missing values: Income = 229; education = 23; Age = 24. Drop them all
# Drop missing data
ss = ss.dropna()


# In[11]:


ss.shape  #after dropping missing values 


# In[12]:


#exploratory analysis 
import altair as alt
alt.Chart(ss).mark_point().encode(
    x="age",
    y="income:N",
    color=alt.Color("sm_li:O").scale(scheme="lightgreyred"),
    column="gender")


# In[13]:


#exploratory analysis 
import altair as alt
alt.Chart(ss).mark_point().encode(
    x="age",
    y="income:N",
    color=alt.Color("sm_li:O").scale(scheme="lightgreyred"),
    column="marital")


# #### 4. Create a target vector (y) and feature set (X)

# In[14]:


#from dataframe ss- y as sm_li; x as income, education, parent, marital, gender, age
y = ss["sm_li"]
X = ss[["income", "education", "parent", "marital","gender","age"]]


# #### 5. Split the data into training and test sets. Hold out 20% of the data for testing. Explain what each new object contains and how it is used in machine learning

# In[15]:


#import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# In[16]:


# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X,y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=987) # set for reproducibility

# X_train contains 80% of the data and contains the features used to predict the target when training the model. 
# X_test contains 20% of the data and contains the features used to test the model on unseen data to evaluate performance. 
# y_train contains 80% of the the data and contains the target that we will predict using the features when training the model. 
# y_test contains 20% of the data and contains the target we will predict when testing the model on unseen data to evaluate performance.


# #### 6. Instantiate a logistic regression model and set class_weight to balanced. Fit the model with the training data.

# In[17]:


# Initialize algorithm 
lr = LogisticRegression(class_weight='balanced').fit(X_train, y_train)


# In[18]:


# Fit algorithm to training data
lr.fit=(X_train, y_train)


# #### 7. Evaluate the model using the testing data. What is the model accuracy for the model? Use the model to make predictions and then generate a confusion matrix from the model. Interpret the confusion matrix and explain what each number means.

# In[19]:


# Make predictions using the model and the testing data
y_pred = lr.predict(X_test)


# In[20]:


# Compare those predictions to the actual test data using a confusion matrix (positive class=1)

#confusion_matrix(y_test, y_pred) and other metrics with classification_report
# Get other metrics with classification_report
print(classification_report(y_test, y_pred))


# #### 8. Create the confusion matrix as a dataframe and add informative column names and index names that indicate what each quadrant represents

# In[21]:


# Confustion matrix as a dataframe
pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns=["Predicted negative", "Predicted positive"],
            index=["Actual negative","Actual positive"]).style.background_gradient(cmap="PiYG")


# #### 9. Aside from accuracy, there are three other metrics used to evaluate model performance: precision, recall, and F1 score. Use the results in the confusion matrix to calculate each of these metrics by hand. Discuss each metric and give an actual example of when it might be the preferred metric of evaluation. After calculating the metrics by hand, create a classification_report using sklearn and check to ensure your metrics match those of the classification_report.

# In[22]:


## Accuracy: TP+TN/(Total instances)
(63+109)/(252)


# In[23]:


## Recall: TP/(TP+FN)
63/(63+21)


# In[24]:


## Precision: TP/(TP+FP)
63/(63+59)


# #### 10. Use the model to make predictions. For instance, what is the probability that a high income (e.g. income=8), with a high level of education (e.g. 7), non-parent who is married female and 42 years old uses LinkedIn? How does the probability change if another person is 82 years old, but otherwise the same?

# In[35]:


# New data for predictions
newdata = pd.DataFrame({
    "income":    [6, 8],     #between 1  to 9
    "education": [8, 5],     #between 1  to 8
    "parent":    [1, 0],     #binary 0, 1
    "marital":   [1, 1],     #binary 0, 1
    "gender":    [0, 1],     #binary 0, 1
    "age":       [40, 62],   #continuous through 98
})
newdata


# In[36]:


# Use model to make predictions
newdata["prediction_sm_li"] = lr.predict(newdata)


# In[37]:


newdata


# ***

# In[49]:


#Making predictions 
##New data for features: age, college, high_income, ideology
person = [8, 5, 1, 1,1,75]

##Predict class, given input features
predicted_class = lr.predict([person])

##Generate probability of positive class (=1)
probs = lr.predict_proba([person])


# In[50]:


##Print predicted class and probability
print(f"Predicted class: {predicted_class[0]}") # 0=Doesn't use LinkedIn, 1=Uses LinkedIN
print(f"Probability that this person is LinkedIn user: {probs[0][1]}")


# ### Part 2 (20%): Deploying the model on Streamlit

# You will now use the model you have developed in part 1 to make live predictions, given a set of inputs set by users of your application. You will use your code and build an application in streamlit. To do so, you must <br>
# move your code into a .py script file, <br>
# create a virtual environment with required packages, <br>
# create a git repository locally, <br>
# create a git repository remotely on GitHub, <br>
# push your local repo (.py script, requirements.txt file with packages, and dataset) to a GitHub repo, and <br>
# host it on Streamlit cloud using your GitHub repo. <br>
# 
# The app should take user input for the features included in the model. The app should return (1) whether the person would be classified as a LinkedIn user or not and (2) the probability that the person uses LinkedIn.

# Two required submissions:
# 
# (1) A notebook with all code and answers to Part 1 submitted in HTML form (Part 1)
# 
# (2) A public URL where we can find and use your app (Part 2)
# 
# In addition to the core questions above, the project grade will be based on the following standard:
# 
# Organization: 20%
# Flow of content: 20%
# Use of visuals/data visualization 20%
# Content and data clarity: 20%
# Depth of content engagement: 20%
# Note: The data used come from Pew and, while publicly available, are to be used for educational purposes only.

# In[ ]:





# In[ ]:





