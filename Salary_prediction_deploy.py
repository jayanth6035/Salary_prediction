#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# Load the dataset
df = pd.read_csv('Salary Prediction of Data Professions.csv')

# Drop rows with any missing values
df = df.dropna()

# Remove duplicate rows
df = df.drop_duplicates()

# Convert 'DOJ' and 'CURRENT DATE' columns to datetime
df['DOJ'] = pd.to_datetime(df['DOJ'])
df['CURRENT DATE'] = pd.to_datetime(df['CURRENT DATE'])

# Calculate total experience in the company
df['Experience'] = (df['CURRENT DATE'] - df['DOJ']).dt.days // 365

# Calculate total experience
df['Total_Experience'] = df['PAST EXP'] + df['Experience']

# Calculate Performance Score
df['Performance_Score'] = df['RATINGS'] * df['Total_Experience']

# Drop specified columns
df = df.drop(columns=['FIRST NAME', 'LAST NAME', 'DOJ', 'CURRENT DATE', 'LEAVES USED', 'LEAVES REMAINING', 'RATINGS','Experience'])

# Encode categorical variables
le = LabelEncoder()
df['SEX'] = le.fit_transform(df['SEX'])
df['DESIGNATION'] = le.fit_transform(df['DESIGNATION'])
df['UNIT'] = le.fit_transform(df['UNIT'])

# Define features and target
X = df.drop(columns=['SALARY'])
y = df['SALARY']

# Splitting of training and testing set into X and y
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the Random Forest Regressor model with best parameters
RFreg = RandomForestRegressor(max_depth=20, min_samples_leaf=4, min_samples_split=10,
                              n_estimators=200, random_state=42)
RFreg.fit(X_train_scaled, y_train)

# Streamlit app
st.title('Salary Prediction of Data Professionals')
st.write('This app predicts the salary of data professions based on input features.')

# Sidebar inputs
st.sidebar.header('Input Features')
sex = st.sidebar.slider('Sex', min_value=0, max_value=1)
Designation = st.sidebar.slider('Designation', min_value=0, max_value=5)
Age = st.sidebar.slider('Age', min_value=21, max_value=50)
Unit = st.sidebar.slider('Unit', min_value=0, max_value=5)
past_exp = st.sidebar.slider('Past Experience', min_value=0, max_value=30)
Total_exp = st.sidebar.slider('Total Experience', min_value=0, max_value=30)
Performance_score = st.sidebar.slider('Performance Score', min_value=0, max_value=30)

# Predict button
predict_button = st.sidebar.button('Predict')

# Check if the Predict button is clicked
if predict_button:
   # Create a dataframe with the selected input features
    input_data = pd.DataFrame({
        'SEX': [sex],
        'DESIGNATION': [Designation],
        'AGE':[Age],
        'UNIT': [Unit],
        'PAST EXP': [past_exp],
        'Total_Experience': [Total_exp],
        'Performance_Score': [Performance_score]
    })

    # Standardize the input data using the same scaler
    input_data_scaled = scaler.transform(input_data)

    # Predict salary
    prediction = RFreg.predict(input_data_scaled)

    # Display the prediction
    st.subheader('Salary Prediction')
    st.write(f'The predicted salary is ${prediction[0]:.2f}')

