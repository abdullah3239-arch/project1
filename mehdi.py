import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

import kagglehub

# Download latest version
df = kagglehub.dataset_download("sanjanchaudhari/employees-performance-for-hr-analytics")

print("Path to dataset files:", df)


def predict_score(features):
    
    return np.random.randint(0, 100)  

# Sidebar for navigation
st.sidebar.title(" Ai experience predictor 🤖")
page = st.sidebar.radio("Select Page:", ["Home", "Predict"])

if page == "Home":
    st.title("✨ Welcome to the Training Experience Predictor ✨")
    name = st.text_input("ENTER YOUR NAME")
    email = st.text_input("ENTER YOUR EMAIL")
    if st.button("Submit"):
        st.success(f"Thank you, {name}! Your information has been submitted.")
        st.balloons()
        

if page == "Predict":
    st.title("Predict Training Score")
    
    # User input for independent features
    age = st.number_input("Age", min_value=20, max_value=60, value=30)
    gender = st.selectbox("Enter your gender", options=["Female", "Male", "Other"])
    service = st.number_input("Service (Years)", min_value=1, max_value=34, value=5)
    training = st.number_input("Training Experience", min_value=1, max_value=10, value=1)
    previous_year_rating = st.number_input("Previous Year Rating", min_value=1, max_value=5, value=3)
    KPIs = st.selectbox("KPIs Met More Than 80%", options=[0, 1])
    awards_won = st.selectbox("Awards Won", options=["Yes", "No"])
    region = st.selectbox("Region", options=["Asia", "Africa", "North America", "South America", "Antarctica", "Europe", "Australia (Oceania)"])
    education = st.selectbox("Education Level", options=["Bachelors", "Masters", "PhD"])

    # Prepare features for prediction
    education_mapping = {"Bachelors": 0, "Masters": 1, "PhD": 2}
    education_num = education_mapping[education]

    features = [
        age,
        service,
        training,
        previous_year_rating,
        KPIs,
        awards_won,
        region,
        education_num
    ]

    # Button to predict score
    if st.button("Predict Score"):
        score = predict_score(features)
        st.success(f"The predicted training score is: {score}")
        st.balloons()
        
def train_model():
    # Preprocessing steps here...
    
   
    df['gender'] = LabelEncoder().fit_transform(df['gender'])
    df['education_encoded'] = LabelEncoder().fit_transform(df['education'])
    
    X = df.drop('KPIs_met_more_than_80', axis=1).values  # Adjust as necessary
    y = df['KPIs_met_more_than_80'].values
    
    
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
    X = np.array(ct.fit_transform(X))

   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Standardization 
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Train stacking classifier model
    base_learners = [('rf_1', DecisionTreeClassifier()), ('rf_2', SVC(probability=True))]
    logreg = LogisticRegression()
    
    stack_clf = StackingClassifier(estimators=base_learners, final_estimator=logreg)
    
    stack_clf.fit(X_train, y_train)
    
    return stack_clf

