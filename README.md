# AI-VISA-STATUS-MODULE-1
import numpy as np
import pandas as pd

#reading csv file with real time dates
df=pd.read_csv("visa_dataset.csv", parse_dates=['Application_Date','Decision_Date']) 
print(df.isnull().sum())

#handling missing values
df['Decision_Date'].fillna(df['Decision_Date'].mode()[0], inplace=True)
df['Processing_Days'].fillna(df['Processing_Days'].mode()[0], inplace=True)
df['Visa_Class'].fillna('Unknown',inplace=True)
df['Gender'].fillna('Unknown',inplace=True)
df['Applicant_Age'].fillna(round(df['Applicant_Age'].median()),inplace=True)
df['Processing_Center'].fillna('Unknown',inplace=True)
df['Case_Status'].fillna('Unknown',inplace=True)

#adding application month as an attribute
df['Application_Month']=df['Application_Date'].dt.month_name()
df.to_csv('visa_dataset.csv',index=False)

#encoding the dataset 
df_encoded=pd.get_dummies(df, columns=['Visa_Type','Visa_Class','Gender','Applicant_Country','Visa_Country','Processing_Center','Case_Status'])
