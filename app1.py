# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 23:58:27 2022

@author: Research-2025
"""

import numpy as np
import pandas as pd
# Python wrapper of firebase
import pyrebase
from datetime import datetime
#from flasgger import Swagger
import streamlit as st 


from PIL import Image
st.set_page_config(page_title='Heart Disease Detection', page_icon = ":heart:")

#app=Flask(__name__)
#Swagger(app)

import tensorflow as tf
model = tf.keras.models.load_model('mymodel')
classifier = model

hide_st_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
st.markdown(hide_st_style, unsafe_allow_html = True)
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


def ContactUs():
    html_temp = """
    <div style="background-color:white;padding:10px">
    <title>Contact Us</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
    <link rel="stylesheet" href="https://www.w3schools.com/lib/w3-theme-teal.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <style>
        body {
                background-image: url("Images/background_contactus.jpg");
                background-repeat: no-repeat;
                background-attachment: fixed;
                background-size: cover;
                margin-top: 8%;
                color: black
                }
            /* Style the container/contact section */
        .container {
                border-radius: 10px;
                background-color: #f2f2f2;
                padding: 10px;
                }
                
            /* Create two columns that float next to eachother */
        .column {
                float: left;
                width: 50%;
                margin-top: 6px;
                padding: 20px;
                }
            
            /* Clear floats after the columns */
        .row:after {
                content: "";
                display: table;
                clear: both;
                }
    </style>
    <body>
            
                    
    <header class="w3-container w3-padding-64 w3-center">
    <h1 class="w3-xxxlarge w3-padding-16">&#128509 Ring us up</h1>
    </header>
                    
    <div class="container w3-center">
    <div style="text-align:center">
    <h2>Contact Us</h2>
    <p>Hi this is Hamza. I'm the Team Lead for this FYP.</p>
    <p>Feel free to connect with us to know how we're doing: <span style='font-size:30px;'>&#127867;</span></p>
    </div>
    <div class="row">
    <strong>&#128241 My Cell-Phone Number:</strong> +92324-8422660<br>
    <strong>&#128241 Usman's Cell-Phone Number:</strong> +92312-1533080<br>
    <strong>&#128241 Shehroz's Cell-Phone Number:</strong> +92304-7769498<br>
    <strong>&#128241 Anas's Cell-Phone Number:</strong> +96278-0332636<br>
     <strong>&#127970 Office Address:</strong>
     <address style="margin-left:15%;"> Electrical Dept,<br> CEME, NUST,<br> Rawalpindi<br></address>
                            <strong>&#128231 Mail us at: </strong><a href="mailto:mmqpak2015@gmail.com">mmqpak2015@gmail.com</a><br>
                            <strong>&#128247 Follow me at: </strong><a
                            href="https://www.instagram.com/knowmeas._.hamza/">knowmeas._.hamza</a><br>
                            <strong>&#128077 LinkedIn: </strong><a
                            href="https://www.linkedin.com/in/hamza-mubashir-66a02a185/">Hamza Mubashir</a>
                            
    </div>
    </div> 
    </div>
                           
                            
    </div>                       
    </body>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
def FrontPage():
    
    html_temp = """
    <div style="background-color:white;padding:10px">
    <title>Contact Us</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
    <link rel="stylesheet" href="https://www.w3schools.com/lib/w3-theme-teal.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <style>
        body {
                background-image: url("Images/background_contactus.jpg");
                background-repeat: no-repeat;
                background-attachment: fixed;
                background-size: cover;
                margin-top: 8%;
                color: black
                }
            /* Style the container/contact section */
        .container {
                border-radius: 10px;
                background-color: #f2f2f2;
                padding: 10px;
                }
                
            /* Create two columns that float next to eachother */
        .column {
                float: left;
                width: 50%;
                margin-top: 6px;
                padding: 20px;
                }
            
            /* Clear floats after the columns */
        .row:after {
                content: "";
                display: table;
                clear: both;
                }
    </style>
    <body>
            
                    
    <header class="w3-container w3-padding-64 w3-center">
    <h1 class="w3-xxxlarge w3-padding-16">&#127973 Take Care of Your Health Anytime Anywhere.</h1>
    </header>
                    
    <div class="container w3-center">
    <div style="text-align:justify">
    <h2>Hey Hello!</h2>
    <p>Welcome to our perception of the future of online healthcare. The future is exciting and we hope that this gives you an idea of what healthcare could be like.</p>
    <p>According to the World Health Organization (WHO), cardiovascular diseases (CVDs) are the leading cause of death globally since they are the cause of 1/3 of global deaths. Research shows that 17.9 million people have died from CVDs in 2019 and a shocking 3/4 of the total casualties are seen in low and middle-income countries. To tackle this situation, it is important that technological advancements of the 21st century are applied to support the health care systems of such developing economies.</p>
    <p>Electrocardiogram (ECG) is extensively employed by cardiologists for observing cardiac health. The main intricacy with the manual analysis of Electrocardiogram signals lies in detecting anomalies in the signal due to the use of time-series data. Still, ECG is a reliable monitoring method for the cardiovascular system. Cardiologists being human can find it complex and challenging to effectively and accurately identify abnormalities. The difficulty increases as the number of patients increase since a lot of these cases are seen in developing countries. This shows that today detecting heart disease through electrocardiogram is of supreme importance to support the overburdened health sector.</p>
    <p>Our application is relying on Deep Learning to effectively, accurately, and timely identify ECG signal abnormalities to showcase and indicate the medical observer so that he is aided in providing the necessary health care. This process includes an interface that provides a patient-physician interface. Patient can then use this interface to upload their ECG and get prompt prediction of whether or not they have abnormal heart condition. In case of abnormal heart condition both the patient and the physician is notified, this calls for an emergency where the physician is responsible to immediately contact the patient and the patient is supposed to immediately visit the hospital. We will try to integrate the developed method onto hardware, if possible, to provide even better support. In this way we have a real time and convenient method for detection that can help detect heart disease at an early stage and prevent patient fatalities. </p>
    </div>
    </div> 
    </div>
                           
                            
    </div>                       
    </body>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

def Admin_Home():
    
    html_temp = """
    <div style="background-color:white;padding:10px">
    <title>Contact Us</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
    <link rel="stylesheet" href="https://www.w3schools.com/lib/w3-theme-teal.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <style>
        body {
                background-image: url("Images/background_contactus.jpg");
                background-repeat: no-repeat;
                background-attachment: fixed;
                background-size: cover;
                margin-top: 8%;
                color: black
                }
            /* Style the container/contact section */
        .container {
                border-radius: 10px;
                background-color: #f2f2f2;
                padding: 10px;
                }
                
            /* Create two columns that float next to eachother */
        .column {
                float: left;
                width: 50%;
                margin-top: 6px;
                padding: 20px;
                }
            
            /* Clear floats after the columns */
        .row:after {
                content: "";
                display: table;
                clear: both;
                }
    </style>
    <body>
            
                    
    <header class="w3-container w3-padding-64 w3-center">
    <h1 class="w3-xxxlarge w3-padding-16">&#127973 Take Care of Your Health Anytime Anywhere.</h1>
    </header>
                    
    <div class="container w3-center">
    <div style="text-align:justify">
    <h2>Hi!</h2>
    <p>Just in case let me take you through a short tutorial of your interface:</p>
    <p>1. Using the "Deep Learning Tab" you can upload your patients ECG personally from your terminal to predict their condition.</p>
    <p>2. Using the "All Patients Record Tab" you can monitor all your patients condition be it normal or abnormal.</p>
    <p>3. Using the "Emergrncy Tab" you will be notified in case any patient has been detected of an abnormal heart condition.</p>
    </div>
    </div> 
    </div>
                           
                            
    </div>                       
    </body>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
def HomePage():
    html_temp = """
    <div style="background-color:white;padding:10px">
    <title>Contact Us</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <link rel="stylesheet" href="https://www.w3schools.com/w3css/4/w3.css">
    <link rel="stylesheet" href="https://www.w3schools.com/lib/w3-theme-teal.css">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/4.7.0/css/font-awesome.min.css">
    <style>
        body {
                background-image: url("Images/background_contactus.jpg");
                background-repeat: no-repeat;
                background-attachment: fixed;
                background-size: cover;
                margin-top: 8%;
                color: black
                }
            /* Style the container/contact section */
        .container {
                border-radius: 10px;
                background-color: #f2f2f2;
                padding: 10px;
                }
                
            /* Create two columns that float next to eachother */
        .column {
                float: left;
                width: 50%;
                margin-top: 6px;
                padding: 20px;
                }
            
            /* Clear floats after the columns */
        .row:after {
                content: "";
                display: table;
                clear: both;
                }
    </style>
    <body>
            
                    
    <header class="w3-container w3-padding-64 w3-center">
    <h1 class="w3-xxxlarge w3-padding-16">&#127973 Take Care of Your Health Anytime Anywhere.</h1>
    </header>
                    
    <div class="container w3-center">
    <div style="text-align:justify">
    <h2>Hi!</h2>
    <p>Just in case let me take you through a short tutorial of your interface:</p>
    <p>1. Using the "Deep Learning Tab" you can upload your ECG personally from your terminal to predict your heart condition.</p>
    <p>2. Using the "Your Record Tab" you can monitor your condition be it normal or abnormal.</p>
    <p>3. Using the "Contact Tab" you can contact your healthcare.</p>
    </div>
    </div> 
    </div>
                           
                            
    </div>                       
    </body>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)

def PersonalRecord():
    html_temp = """
    <div class="container w3-center">
    <div style="text-align:justify">
    <h2>Below is your Record</h2>
    <p>Records show your condition periodically with time stamps.</p>

    </div>
    </div> 
    </div>
                           
                            
    </div>                       
    </body>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    if st.button("Show Record"):
        all_posts = db.child(user['localId']).child("Posts").get()
        if all_posts.val() is not None:
            for Posts in reversed(all_posts.each()):
                #st.write(Posts.key()) #Morty
                #st.code(Posts.val(), language = '')
                st.success(Posts.val())

def AdminRecord():
    html_temp = """
    <div class="container w3-center">
    <div style="text-align:justify">
    <h2>Below are all Patient Records</h2>
    <p>You can use the selection to go through patients records.</p>
    <p>Records show periodic patient condition with time stamps.</p>

    </div>
    </div> 
    </div>
                           
                            
    </div>                       
    </body>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    
    all_users = db.get()
    res = []
    
    # store all the users handle name
    for users_handle in all_users.each():
        k = users_handle.val()["Handle"]
        res.append(k)
    # Total users
    nl = len(res)
    st.write('Total patients here: '+ str(nl))
    
    # Allow the user to choose which other user he/she wants to see
    choice = st.selectbox('My Patients', res)
    push = st.button('Show Profile')
    
    # Show the choosen Profile
    if push:
        for users_handle in all_users.each():
            k = users_handle.val()["Handle"]
            if k == choice:
                lid = users_handle.val()["ID"]
        
                handlename = db.child(lid).child("Handle").get().val()
                
                st.markdown(handlename, unsafe_allow_html = True)
                
                # For all posts
                all_posts = db.child(lid).child("Posts").get()
                if all_posts.val() is not None:
                    for Posts in reversed(all_posts.each()):
                        st.code(Posts.val(), language = '')
                else:
                    st.success("No record found")
def emergency():
    html_temp = """
    <div class="container w3-center">
    <div style="text-align:justify">
    <h2>All Emergency Cases are notified here.</h2>
    <p>Incase Patient is predicted as abnormal, notify the health department for an emergency and contact the patient to immediately report to the hospital.</p>

    </div>
    </div> 
    </div>
                           
                            
    </div>                       
    </body>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    all_users = db.get()
    res = []
    check = 0
    for users_condition in all_users.each():
        k = users_condition.val()["Condition"]
        res.append(k)
    for users_handle in all_users.each():
        k = users_condition.val()["Condition"]
        if k == "Abnormal":
            check = 1
            lid = users_condition.val()["ID"]
    
            handlename = db.child(lid).child("Handle").get().val()
            
            disp = 'Emergency Case: For Patient {} the condition is abnormal '.format(handlename)
            st.error(disp)
    if check == 0:
        disp = 'All Patients are healthy'
        st.success(disp)
    
                     
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
        condition = (display)
        db.child(user['localId']).child("Condition").set(display)
        Patient_info = 'For Patient {} with ID {}. '.format(name,ID)
        Disease_info = 'Patients Heart Condition is {}'.format(display)
        st.success(Patient_info)
        st.success(Disease_info)
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        post = {'Post' : Patient_info + Disease_info,
                'Timestamp' : dt_string,
                'Condition' : condition}
        results = db.child(user['localId']).child("Posts").push(post)
        
        all_posts = db.child(user['localId']).child("Posts").get()
        if all_posts.val() is not None:
            for Posts in reversed(all_posts.each()):
                #st.write(Posts.key()) #Morty
                st.code(Posts.val(), language = '')
        
    if st.button("About"):
        st.text("Lets Learn")
        st.text("Built with Streamlit")


# Configuration Key
firebaseConfig = {
  'apiKey': "AIzaSyBwV_ya3J5MMQte1rPM5AJFkTMSoEmVv7A",
  'authDomain': "myapphamza3.firebaseapp.com",
  'projectId': "myapphamza3",
  'databaseURL':"https://myapphamza3-default-rtdb.firebaseio.com/",
  'storageBucket': "myapphamza3.appspot.com",
  'messagingSenderId': "982785824696",
  'appId': "1:982785824696:web:5345e02e401c4dfb6f614d",
  'measurementId': "G-ZZ84M0E84B"
};

# Firebase Authentication
firebase = pyrebase.initialize_app(firebaseConfig)
auth = firebase.auth()

# Database
db = firebase.database()
storage = firebase.storage()

st.sidebar.title("Heart Disease Detection App")

# Authentication
choice = st.sidebar.selectbox('Login/Signup',['Login','Sign Up'])
email = st.sidebar.text_input('Please enter your email address')
password = st.sidebar.text_input('Please enter your password', type = 'password')

if choice == 'Sign Up':
    login = 0
    handle = st.sidebar.text_input('Please input your name', value = 'Name')
    submit = st.sidebar.button("Create my account")
    
    if submit:
        if handle == "Doctor":
            st.info('This name can not be taken. Please sign up using another name.')
        else:
            user = auth.create_user_with_email_and_password(email, password)
            st.success('Your account has been created successfully!')
            st.balloons()
            # Sign In
            user = auth.sign_in_with_email_and_password(email, password)
            db.child(user['localId']).child("Handle").set(handle)
            db.child(user['localId']).child("ID").set(user['localId'])
            st.info('Login via login drop down menu')

if choice == 'Login':
    login = st.sidebar.checkbox('Login')
    if login:
        user = auth.sign_in_with_email_and_password(email, password)
        handle = db.child(user['localId']).child("Handle").get().val()
        st.title('Welcome '+handle)
        if handle == "Doctor":
            admin_display = 1
        else:
            admin_display = 0
        

#with st.sidebar:
if login == 1:
    if admin_display == 0:
        selected = option_menu(
            menu_title = "Main Menu",
            options = ["Home", "Deep Learning", "Your Record" ,"Contact"],
            icons = ["house", "binoculars", "clipboard", "envelope"],
            menu_icon = "cast",
            default_index = 0,
            orientation ="horizontal"
            )
    if admin_display == 1:
        selected = option_menu(
            menu_title = "Main Menu",
            options = ["Admin Home", "Deep Learning", "All Patient Record", "Emergency"],
            icons = ["house", "binoculars", "clipboard", "exclamation-triangle"],
            menu_icon = "cast",
            default_index = 0,
            orientation ="horizontal"
            )
    
if login == 0:
    selected = option_menu(
        menu_title = "Main Menu",
        options = ["Welcome", "Contact"],
        icons = ["house","envelope"],
        menu_icon = "cast",
        default_index = 0,
        orientation ="horizontal"
        )
    
def main():
    if selected == "Deep Learning":
        deeplearning()
    if selected == "Home":
        HomePage()
    if selected == "Contact":
        ContactUs()
    if selected == "Your Record":
        PersonalRecord()
    if selected == "Welcome":
        FrontPage()
    if selected == "Admin Home":
        Admin_Home()
    if selected == "All Patient Record":
        AdminRecord()
    if selected == "Emergency":
        emergency()
    
if __name__=='__main__':
    main()
