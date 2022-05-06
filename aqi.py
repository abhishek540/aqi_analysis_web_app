# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("AQI Analysis")
st.write("Dataset")



df = pd.read_csv("city_day.csv",sep=",")

st.write("Top 5 Dataset",df.head())

st.write("Description of Dataset",df.describe())

#Separate the data into features and target
df=df[df['Date'] >= ('2020-01-01')]
df.reset_index(drop=True,inplace=True)

#filling null Value
df['PM2.5']=df['PM2.5'].fillna(df['PM2.5'].mean())
df['PM10']=df['PM10'].fillna(df['PM10'].mean())
df['NO']=df['NO'].fillna(df['NO'].mean())
df['NO2']=df['NO2'].fillna(df['NO2'].mean())
df['NOx']=df['NOx'].fillna(df['NOx'].mean())
df['NH3']=df['NH3'].fillna(df['NH3'].mean())
df['CO']=df['CO'].fillna(df['CO'].mean())
df['SO2']=df['SO2'].fillna(df['SO2'].mean())
df['O3']=df['O3'].fillna(df['O3'].mean())
df['Benzene']=df['Benzene'].fillna(df['Benzene'].mean())
df['Toluene']=df['Toluene'].fillna(df['Toluene'].mean())
df['Xylene']=df['Xylene'].fillna(df['Xylene'].mean())
df['AQI']=df['AQI'].fillna(df['AQI'].mode()[0])
df['AQI_Bucket']=df['AQI_Bucket'].fillna('Moderate')

# how much is the average amount of pollution in each city stations
most_polluted = df[['City', 'AQI', 'PM10', 'CO']].groupby(['City']).mean().sort_values(by = 'AQI', ascending = False)
st.write("Most Polluted City",most_polluted)

df1=df.copy()
df1['Vehicle_Pollution_content']=df1['PM2.5']+df1['PM10']+df1['NO']+df1['NOx']+df1['NH3']+df1['CO']
df1['Industry_pollutants']=df1['SO2']+df1['O3']+df1['Benzene']+df1['Toluene']+df1['Xylene']
df1.drop(['PM2.5','PM10','NO','NO2','NOx','NH3','CO','SO2','O3','Benzene','Toluene','Xylene'],axis=1,inplace=True)

df1['Day_date']=pd.to_datetime(df1['Date'],format='%Y/%m/%d').dt.day
df1['month_date']=pd.to_datetime(df1['Date'],format='%Y/%m/%d').dt.month
df1.drop(['Date'],axis=1,inplace=True)


outliers=df1.loc[df1['Vehicle_Pollution_content'] > (1000)]
outliers=df1.loc[df1['Industry_pollutants']>(800)]

df1.drop(['AQI_Bucket'],axis=1,inplace=True)

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

df1=pd.get_dummies(df1,drop_first=True)

X=df1.drop(['AQI'],axis=1)
y=df1['AQI']

import sklearn
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

from sklearn.metrics import classification_report,accuracy_score
from sklearn.ensemble import ExtraTreesRegressor
model3=ExtraTreesRegressor()
model3.fit(X_train,y_train)
accuracy = model3.score(X_test,y_test)
st.write("Accuracy score of the model is ",accuracy)
y_test.reset_index(inplace = True,drop =True )

#AQI = st.text_input('AQI',)
PM2_5 = st.text_input('PM2_5',79.57)
PM10 = st.text_input('PM10',131.62)
NO = st.text_input('NO',3.78)
NO2 = st.text_input('NO2',12.64)
NOx = st.text_input('Nox',8.99)
NH3 = st.text_input('NH3',0)
CO = st.text_input('CO',3.78)
SO2 = st.text_input('SO2',27.70)
O3 = st.text_input('O3',23.67)
Benzene = st.text_input('Benzene',4.21)
Toluene = st.text_input('Toluene',31.42)
Xylene = st.text_input('Xylene',2.52)
AQI_Bucket = st.text_input('AQI_Bucket',"Poor")
city_list = [ 'Aizawl', 'Amaravati', 'Amritsar', 'Bengaluru',
       'Bhopal', 'Brajrajnagar', 'Chandigarh', 'Chennai', 'Coimbatore',
       'Delhi', 'Ernakulam', 'Gurugram', 'Guwahati', 'Hyderabad',
       'Jaipur', 'Jorapokhar', 'Kochi', 'Kolkata', 'Lucknow', 'Mumbai',
       'Patna', 'Shillong', 'Talcher', 'Thiruvananthapuram',
       'Visakhapatnam']
city  = st.selectbox("Select City", city_list)

Vechile_Pollution_content = float(PM2_5)+float(PM10)+float(NO)+float(NOx)+float(NH3)+float(CO)
Industry_pollutants = float(SO2)+float(SO2)+float(Benzene)+float(Toluene)+float(Xylene)
Day_date = st.text_input("Day of the date",1)

month_date = st.text_input("Month of the date",1)

list_cities = ['City_Aizawl', 'City_Amaravati', 'City_Amritsar',
       'City_Bengaluru', 'City_Bhopal', 'City_Brajrajnagar', 'City_Chandigarh',
       'City_Chennai', 'City_Coimbatore', 'City_Delhi', 'City_Ernakulam',
       'City_Gurugram', 'City_Guwahati', 'City_Hyderabad', 'City_Jaipur',
       'City_Jorapokhar', 'City_Kochi', 'City_Kolkata', 'City_Lucknow',
       'City_Mumbai', 'City_Patna', 'City_Shillong', 'City_Talcher',
       'City_Thiruvananthapuram', 'City_Visakhapatnam']

dict_data = {"Vechile_Pollution_content":Vechile_Pollution_content,
             "Industry_pollutants":Industry_pollutants,
             "Day_date":Day_date,
             "month_data":month_date,
             "City_Aizawl":0, 
             'City_Amaravati':0,
             'City_Amritsar':0,
             'City_Bengaluru':0,
             'City_Bhopal':0,
             'City_Brajrajnagar':0, 
             'City_Chandigarh':0,
             'City_Chennai':0,
             'City_Coimbatore':0,
             'City_Delhi':0, 
             'City_Ernakulam':0,
             'City_Gurugram':0,
             'City_Guwahati':0,
             'City_Hyderabad':0,
             'City_Jaipur':0,
             'City_Jorapokhar':0,
             'City_Kochi':0,
             'City_Kolkata':0,
             'City_Lucknow':0,
             'City_Mumbai':0,
             'City_Patna':0,
             'City_Shillong':0,
             'City_Talcher':0,
             'City_Thiruvananthapuram':0,
             'City_Visakhapatnam':0}

selected_city = "City_"+city

dict_data[selected_city] = 1
dfx = pd.DataFrame(dict_data, index = [0])
st.write("Input Data",dfx)

st.write("Predicted AQI",model3.predict(dfx))







