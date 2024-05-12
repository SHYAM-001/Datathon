# -*- coding: utf-8 -*-
"""criminal_preprocessed_data.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/157RhV7lHvPmMgHJBpLg_GtYAdRsiiez6
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_accused=pd.read_csv('/content/AccusedData.csv')

df_accused.head()

df_accused.shape

df_accused.duplicated().sum()

df_accused.drop_duplicates(inplace=True)

df_accused.isnull().sum()

df_accused.dtypes

#Generate date from year and month
df_accused['date'] = pd.to_datetime(df_accused[['Year', 'Month']].assign(day=1))

df_accused.columns

df_accused=df_accused.drop(columns=['Year','Month'])

df_accused.columns

#given the age, dob is not required
df_accused.drop(columns=['DOB'],inplace=True)

df_accused.columns

#fillage
plt.figure(figsize=(7,7))
df_accused.boxplot('age')

df_accused['age'].mean()

df_accused['age'].median()

df_accused['age'].fillna(df_accused['age'].median(),inplace=True)

df_accused.isnull().sum()

#profession and sex
from scipy.stats import chi2_contingency

# Contingency table between 'Profession' and 'Other_Category'
columns_to_test = ['Caste', 'Sex', 'PresentState', 'PermanentState']
p_values = {}
chi2_values={}
for column in columns_to_test:
    contingency_table = pd.crosstab(df_accused['Profession'], df_accused[column])
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    p_values[column] = p
    chi2_values[column]=chi2

print(p_values)
print(chi2_values)

#sex
df_accused.dropna(subset=['Sex'], inplace=True)

df_accused.shape

df_accused.dropna(subset=['Caste'], inplace=True)

mode_of_os=df_accused.pivot_table(values='Profession',columns='Caste',aggfunc=(lambda x:x.mode()))
missing_values=df_accused['Profession'].isnull()
df_accused.loc[missing_values,'Profession'] = df_accused.loc[missing_values,'Caste'].apply(lambda x: mode_of_os[x].Profession)

df_accused.isnull().sum()

df_accused.columns

#person name
df_accused.dropna(subset=['Person_Name'], inplace=True)

df_accused.shape

len(df_accused['Person_Name'].unique())

df_accused.drop(columns=['AccusedName'],inplace=True)

df_accused.isnull().sum()

df_accused['Nationality_Name'].value_counts()

len(df_accused['PresentAddress'].unique())

df_accused['PermanentAddress'].value_counts()

len(df_accused['PermanentAddress'].unique())

df_accused.drop(columns=['PermanentAddress'],inplace=True)

df_accused.drop(columns=['PresentAddress'],inplace=True)

df_accused.isnull().sum()

df_accused.dropna(subset=['PresentCity'],inplace=True)
df_accused.dropna(subset=['PresentState'],inplace=True)

df_accused[df_accused['Nationality_Name'] == 'Myanmar']

df_accused['Person_No'].value_counts()

df_accused.dropna(subset=['Person_No'], inplace=True)

df_accused.isnull().sum()

df_accused.shape

filename='Accused_fina-1l.xlsx'
df_accused.to_excel('Accused_final_1.xlsx', index=False)

df=pd.read_excel('/content/Accused_final_1.xlsx')

