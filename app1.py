# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 23:58:27 2022

@author: Research-2025
"""

import numpy as np
import pandas as pd
#from flasgger import Swagger
import streamlit as st 

from PIL import Image
st.set_page_config(page_title='Heart Disease Detection', page_icon = ":heart:")

#app=Flask(__name__)
#Swagger(app)

import tensorflow as tf
model = tf.keras.models.load_model('mymodel')
classifier = model


#@app.route('/')
def welcome():
    return "Welcome All"

#@app.route('/predict',methods=["Get"])
def predict_heart_disease(df):
    
    """Let's Authenticate the Banks Note 
    This is using docstrings for specifications.
    ---
    parameters:  
      - name: variance
        in: query
        type: number
        required: true
      - name: skewness
        in: query
        type: number
        required: true
      - name: curtosis
        in: query
        type: number
        required: true
      - name: entropy
        in: query
        type: number
        required: true
    responses:
        200:
            description: The output values
        
    """
   
    prediction=classifier.predict(df)
    print(prediction)
    return prediction

from streamlit_option_menu import option_menu

# 1. as horizontal menu
#with st.sidebar:
selected = option_menu(
    menu_title = "Main Menu",
    options = ["Home", "Deep Learning", "Record" ,"Contact"],
    icons = ["house", "binoculars", "clipboard", "envelope"],
    menu_icon = "cast",
    default_index = 0,
    orientation ="horizontal"
    )

if selected == "Home":
    st.title(f"Write Home Text Below")

def deeplearning():
    st.title("Heart Disease Detection")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Using Deep Learning</h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    name = st.text_input("Patient Name","")
    ID = st.text_input("Patient ID","")
    #curtosis = st.text_input("curtosis","Type Here")
    #entropy = st.text_input("entropy","Type Here")
    st.subheader("Upload your ECG below")
    data_file = st.file_uploader("Upload CSV", type=["csv"])
    if data_file is not None:
        st.write(type(data_file))
        file_details = {"filename":data_file.name,
                        "filetype":data_file.type, "filesize":data_file.size}
        st.write(file_details)
        df = pd.read_csv(data_file)
        st.dataframe(df)
        print(df)
    
    result=""
    if st.button("Predict"):
        #result=predict_note_authentication(variance,skewness,curtosis,entropy)
        result=predict_heart_disease(df)
        if result == 0:
            display = "Abnormal"
        elif result == 1:
            display = "Normal"
        st.success('For Patient {} with ID {}'.format(name,ID))
        st.success('Patients Heart Condition is {}'.format(display))
    if st.button("About"):
        st.text("Lets Learn")
        st.text("Built with Streamlit")

def main():
    if selected == "Deep Learning":
        deeplearning()
    

if __name__=='__main__':
    main()