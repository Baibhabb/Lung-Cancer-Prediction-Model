import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os


model_files = {
    "Logistic Regression": "ML_model/Logistic_Regression_model.pkl",
    "Decision Tree": "ML_Model/Decision_Tree_model.pkl",
    "Random Forest": "ML_Model/Random_Forest_model.pkl",
    "KNN": "ML_Model/KNN_model.pkl",
    "SVM": "ML_Model/SVM_model.pkl",
    "Naive Bayes (Classification)": "ML_Model/Naive_Bayes_model.pkl"
}

loaded_models = {}
for name, path in model_files.items():
    if os.path.exists(path):
        loaded_models[name] = joblib.load(path)


st.title(" Lung Cancer Prediction System")
st.write("Enter the details below to predict the price.")

# User Inputs
GENDER = st.selectbox("Gender?",["M","F"])
AGE=st.number_input("AGE?",min_value=1,max_value=90,value=30)
SMOKING = st.selectbox("SMOKING?", ["YES", "NO"])
YELLOW_FINGERS = st.selectbox("YELLOW FINGERS?", ["YES", "NO"])
ANXIETY = st.selectbox("ANXIETY?", ["YES", "NO"])
PEER_PRESURE = st.selectbox("PEER PRESURE?", ["YES", "NO"])
CHORNIC_DISEASE = st.selectbox("Chornic disease?", ["YES", "NO"])
FATIGUE = st.selectbox("Fatigue?", ["YES", "NO"])
ALLERGY = st.selectbox("Allergy?", ["YES", "NO"])
WHEEZING=st.selectbox("Wheezing?",["YES","NO"])
ALCOHOL_CONSUMING=st.selectbox("Alcohol consuming?",["YES","NO"])
COUGHING=st.selectbox("Coughinf?",["YES","NO"])
SHORTNESS_OF_BREATH=st.selectbox("Shortness of breath?",["YES","NO"])
SWALLOWING_DIFFICULTY=st.selectbox("Swollowing Difficulty?",["YES","NO"])
CHEST_PAIN=st.selectbox("Chest Pain?",["YES","NO"])


# Convert to DataFrame
input_data = pd.DataFrame([[
    GENDER,AGE,SMOKING,YELLOW_FINGERS,ANXIETY,
    PEER_PRESURE,CHORNIC_DISEASE,ALLERGY,FATIGUE,WHEEZING,
    ALCOHOL_CONSUMING,COUGHING,SHORTNESS_OF_BREATH,SWALLOWING_DIFFICULTY,CHEST_PAIN
]], columns=[
    "GENDER","AGE","SMOKING","YELLOW_FINGERS","ANXIETY",
    "PEER_PRESSURE","CHORNIC_DISEASE","ALLERGY ","FATIGUE ","WHEEZING",
    "ALCOHOL_CONSUMING","COUGHING","SHORTNESS_OF_BREATH","SWALLOWING DIFFICULTY","CHEST_PAIN"
])


st.subheader("Choose a Model for Prediction")

model_choice = st.selectbox("Select Model", list(loaded_models.keys()))

if st.button("Predict"):
    model = loaded_models[model_choice]
    prediction = model.predict(input_data)

    if "Naive Bayes" in model_choice:
        if prediction[0] == 1:
         st.success(f"CHANGE FOR CANCER: **YES**")
        else:
         st.success(f"CHANGE FOR CANCER: **NO**")
    else:
        if prediction[0] == 1:
         st.success(f"CHANGE FOR CANCER: **YES**")
        else:
         st.success(f"CHANGE FOR CANCER: **NO**")