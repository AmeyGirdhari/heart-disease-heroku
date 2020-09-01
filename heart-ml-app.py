import pandas as pd
import numpy as np
import streamlit as st
from sklearn.linear_model import LogisticRegression


st.title('Heart Disease Prediction App')

st.write('this app predicts the person is having ** heart disease ** or not ')

st.header('User Input Parameters')

def user_input_features():
    sex      =st.number_input('Sex')
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

sk=pd.read_csv('heart-disease.csv')

sk=sk[['sex','cp','trestbps','restecg','thalach','exang','oldpeak','slope','ca','thal','target']]

x=sk.iloc[:,0:10].values

y=sk.iloc[:,-1].values

np.random.seed(97)
    
model=LogisticRegression(solver='liblinear',C=0.23357214690901212)
       
model.fit(x,y)
predictions=model.predict(df)
kl=''
if predictions==0:
	predictions='not heart-disease'
else:
	predictions='having heart-disease'

if st.button('Predict'):
	kl=predictions
st.success(kl)

if st.button("About"):
	st.text('Developed by Amey Girdhari')
	st.text('Built with streamlit')