import pandas as pd
import numpy as np
import streamlit as st
import pickle
from sklearn.linear_model import LogisticRegression


st.title('Heart Disease Prediction ')

st.write('This app predicts the person is having ** heart disease ** or not based on several medical conditions')

from PIL import Image
image = Image.open('heart.png')

st.image(image)

st.write('''
  In 21'st Century heart disease is very common problems in human beings.
  it hampers the living life of the human being. for healthy life humans needed regulary health checkups.
  for heart disease checkup this app directly predicts the results based on users input data.

  following are the descriptions of terminologies used in this app 

 * cp= chest pain type.
   (1 = typical angina,2 = atypical angina,3 = non-anginal pain,4 = asymptomatic)

 * trestbps = resting blood pressure. (in mm Hg)

 * restecg = resting electrocardiographic results. 
   (0: normal,1= having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV)
    2=showing probable or definite left ventricular hypertrophy by Estes' criteria)

 * thalach = maximum heart rate achieved.

 * exang = exercise induced angina (1 = yes; 0 = no) 

 * oldpeak = ST depression induced by exercise relative to rest 

 * slope: the slope of the peak exercise ST segment.
   (1 = upsloping,2 = flat,3 = downsloping)

 * ca: number of major vessels (0-3) colored by flourosopy.

 * thal: 3 = normal; 6 = fixed defect; 7 = reversable defect         

  ''')

st.header('User Input Parameters')

def user_input_features():
    sex      =st.text_input('Sex')
    cp       =st.number_input	('Cp')
    trestbps =st.number_input('Trestbps')
    restecg  =st.number_input('Restecg')
    thalach  =st.number_input('Thalach')
    exang    =st.number_input('Exang')
    oldpeak  =st.number_input('Oldpeak')
    slope    =st.number_input('Slope')
    ca       =st.number_input('Ca')
    thal     =st.number_input('Thal')
    
    data={ 'sex':sex,
           'cp':cp,
           'trestbps':trestbps,
           'restecg':restecg,
           'thalach':thalach,
           'exang':exang,
           'oldpeak':oldpeak,
           'slope':slope,
           'ca':ca,
           'thal':thal}
    features=pd.DataFrame(data,index=[0])
    return features
    

df= user_input_features()

st.subheader('User Input Parameters')

st.write(df)

transform={
  'sex':{'male':1,'female':0}
}

for col in transform:
    df.loc[:, col] = df.loc[:, col].map(transform[col], na_action='ignore')

    
model=pickle.load(open('heart.pkl','rb'))       

predictions=model.predict(df)
kl=''
if predictions==0:
	predictions='''User has not heart-disease.keep the following habit
  
                 1)Don't smoke or use tobacco.
                 2)Get moving: Aim for at least 30 to 60 minutes of activity daily.
                 3)Eat a heart-healthy diet.
                 4)Maintain a healthy weight.'''
else:
	predictions='''User has heart-disease.strictly follow the below things

                 1)Stop smoking.
                 2)Control your blood pressure.
                 3)Check your cholesterol.
                 4)Keep diabetes under control.'''

if st.button('Predict'):
	kl=predictions
st.success(kl)

if st.button("About"):
	st.text('Developed by Amey Girdhari')
	st.text('Built with streamlit')