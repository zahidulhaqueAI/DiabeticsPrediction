
import streamlit as st
import pandas as pd
import numpy as np
import pickle 
from sklearn.preprocessing import StandardScaler

# load the pickle file
def load_model(model_name):
    with open(model_name, 'rb') as file:
        model, scaler = pickle.load(file)
    return model, scaler

# processing for model
def prepossing_input_data(data, scaler):
    data['sex'] = data['sex'].map({'Male' : 1, "Female" : 2})
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
    st.title('**Diabetics Performance Predicition**')
    st.title('Diabetics Performance Predicition')
    st.write('Enter your data to get a prediction for your performance')
    
    # age	sex	bmi	bp	s1	s2	s3	s4	s5	s6	
    #age = st.number_input("age", min_value=5, max_value=100, value=5)
    

# call the main
if __name__ == '__main__':
    main()