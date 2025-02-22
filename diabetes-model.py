
import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from sklearn.preprocessing import StandardScaler

# connection to mongo db
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi

uri = "mongodb+srv://zahid:zahid1234@cluster0.sltur.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# create database
db = client['diabetes']

# create collection
collection = db['diabetes_pred']

# load the pickle file
def load_model(model_name):
    with open(model_name, 'rb') as file:
        model, scaler = pickle.load(file)
    return model, scaler

# processing for model
def prepossing_input_data(data, scaler):
    df = pd.DataFrame([data])
    df_transformed = scaler.transform(df)
    return df_transformed

# prediction for model
def predict_data(data, model_name):
    model, scaler = load_model(model_name)
    processed_data = prepossing_input_data(data, scaler)
    prediction = model.predict(processed_data)
    return prediction

def main():
    st.title("Diabetes Progression Prediction")
    model_choice = st.sidebar.radio("Choose Model", [ "Ridge Regression", "Lasso Regression"])
    
    st.header("Input Parameters")
    
    age = st.slider("Age of Patient", -0.2, 0.2, 0.0, 0.01)
    sex = st.slider("Sex (normalized)", -0.1, 0.1, 0.0, 0.01)
    bmi = st.slider("BMI (Body Mass Index)", -0.1, 0.2, 0.0, 0.01)
    bp = st.slider("Blood Pressure", -0.1, 0.2, 0.0, 0.01)
    s1 = st.slider("Serum Level 1", -0.2, 0.2, 0.0, 0.01)
    s2 = st.slider("Serum Level 2", -0.2, 0.2, 0.0, 0.01)
    s3 = st.slider("Serum Level 3", -0.2, 0.2, 0.0, 0.01)
    s4 = st.slider("Serum Level 4", -0.2, 0.2, 0.0, 0.01)
    s5 = st.slider("Serum Level 5", -0.2, 0.2, 0.0, 0.01)
    s6 = st.slider("Serum Level 6", -0.2, 0.2, 0.0, 0.01)
    
    # age	sex	bmi	bp	s1	s2	s3	s4	s5	s6	
    user_data = {
            "age": age,
            "sex": sex,
            "bmi": bmi,
            "bp": bp,
            "s1": s1,
            "s2": s2,
            "s3": s3,
            "s4": s4,
            "s5": s5,
            "s6": s6
        }
    
    if st.button("Predict") :
        # model selection
        model_name = 'diabetes_ridge_model.pkl' if model_choice == 'Ridge Regression' else 'diabetes_lasso_model.pkl'

        # Make Prediction
        prediction = predict_data(user_data, model_name)
        predicted_value = round(float(prediction[0]) ,4)  # Convert to float
        
        st.success(f'Your prediction result is {prediction}')
        
        # add to the collection
        # get the model selected
        model_sel = 'Ridge' if model_name == 'diabetes_ridge_model.pkl' else 'Lasso'
        
        user_data['prediction'] = predicted_value
        user_data['Model Selected'] = model_sel
        
        collection.insert_one(user_data)

# call the main
if __name__ == '__main__':
    main()