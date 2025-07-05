import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OrdinalEncoder


#loading train model
model=tf.keras.models.load_model('model.h5')


#load encoder , one hot encoder and scaler
with open("one_hot_encoder_geo.pkl", 'rb') as file:
    one_hot_encoder_geo = pickle.load(file)
with open("label_encode_gender.pkl", 'rb') as file:
    label_encode_gender = pickle.load(file)
with open("scaler.pkl", 'rb') as file:
    scaler= pickle.load(file)

#streamlit app
st.title("Customer Churn prediction")


#userinput
geogprahy = st.selectbox("Geogprahy", one_hot_encoder_geo.categories_[0])
gender= st.selectbox("Gender", label_encode_gender.classes_)
age= st.slider("Age", 18, 75)
balance= st.number_input("Balance")
creditscore= st.number_input("Credit Score")
estimated_salary= st.number_input("Estimated Salary")
tenure= st.slider("Tenure", 0, 5)
num_of_products= st.slider("num of prod", 1, 10)
has_cr_card = st.selectbox("Has c.card", [0,1])
is_active_member= st.selectbox("Active member", [0,1])

#prepare input data
input_data=pd.DataFrame( {
    "CreditScore" : [creditscore],
    "Gender" : [label_encode_gender.transform([gender])[0]],
    "Age" : [age],
    "Tenure" : [tenure],
    "Balance" : [balance],
    "NumOfProducts" : [num_of_products],
    "HasCrCard" : [has_cr_card],
    "IsActiveMember": [is_active_member],
    "EstimatedSalary": [estimated_salary]
}
)
#one hot encoder
geo_encoded = one_hot_encoder_geo.transform([[geogprahy]]).toarray()
geo_encoded_df= pd.DataFrame(geo_encoded, columns=one_hot_encoder_geo.get_feature_names_out(["Geography"]))

input_data= pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

input_data_scaled= scaler.transform(input_data)

#prediction
prediction= model.predict(input_data_scaled)
prediction_prob= prediction[0][0]

if prediction_prob>0.5:
   st.write("Churn successful")
else :
    st.write("No churn")

