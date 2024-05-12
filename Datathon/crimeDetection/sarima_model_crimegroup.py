# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

df =pd.read_csv(r"crimeDetection\ML_models\Preprocessed_FIR_Data1.csv")
df.head()

df.drop(['Unnamed: 0'],axis=1,inplace=True)

df.head()

data.columns

df.dropna(inplace=True)

df['Offence_From_Date'] = pd.to_datetime(df['Offence_From_Date'])

df.index = pd.DatetimeIndex(df.Offence_From_Date)

# top 3 most occuring crime types

most_common_crimes = df['CrimeGroup_Name'].value_counts().head(3)

print(f"Top 3 most occurring crime types:")
for crime_type, count in most_common_crimes.items():
  print(f"\t{crime_type}: {count}")

df_theft = df[df['CrimeGroup_Name'] == 'THEFT']
df_missing_person = df[df['CrimeGroup_Name'] == 'MISSING PERSON']
df_motor_vehicle_accidents_non_fatal = df[df['CrimeGroup_Name'] == 'MOTOR VEHICLE ACCIDENTS NON-FATAL']

df_missing_person.head()

df_theft = pd.DataFrame(df_theft.resample('M').size().reset_index())

df_theft

df_theft.columns

df_missing_person = pd.DataFrame(df_missing_person.resample('M').size().reset_index())

df_missing_person

df_motor_vehicle_accidents_non_fatal= pd.DataFrame(df_motor_vehicle_accidents_non_fatal.resample('M').size().reset_index())

df_motor_vehicle_accidents_non_fatal

df_theft.rename(columns={0: "count", "Offence_From_Date": "month"}, inplace=True)
df_missing_person.rename(columns={0: "count", "Offence_From_Date": "month"}, inplace=True)
df_motor_vehicle_accidents_non_fatal.rename(columns={0: "count", "Offence_From_Date": "month"}, inplace=True)

df_theft.head()

df_missing_person.head()

df_motor_vehicle_accidents_non_fatal.to_csv('crimeDetection\ML_models\motor_vehicle.csv')

df_missing_person.to_csv('crimeDetection\ML_models\missing_person.csv')

df_theft.to_csv('crimeDetection\ML_models\theft.csv')

df_theft.columns

import statsmodels.api as sm
import plotly.graph_objs as go

df_theft.set_index('month', inplace=True)
df_missing_person.set_index('month', inplace=True)
df_motor_vehicle_accidents_non_fatal.set_index('month', inplace=True)

import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX
from plotly.subplots import make_subplots

# Define the SARIMA model
model = SARIMAX(df_theft["count"], order=(2, 1, 2), seasonal_order=(1, 0, 1, 12))


results = model.fit()

# Create a plotly subplot
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Original Data", "Predicted Future Values"), vertical_spacing=0.3)

# Plot the original data
fig.add_trace(go.Scatter(x=df_theft.index, y=df_theft["count"], name="Original Data"), row=1, col=1)

forecast_horizon = 12
forecast = results.forecast(steps=forecast_horizon)

# Plot the predicted future values
fig.add_trace(go.Scatter(x=pd.date_range(start=df_theft.index[-1], periods=forecast_horizon + 1, freq="M"), y=forecast, name="Predicted Future Values"), row=2, col=1)

# Update the layout and show the plot
fig.update_layout(height=900, title_text="SARIMA Model for Theft Data")
fig.show()

import pandas as pd
import pickle

with open('crimeDetection\ML_models\sarima_model_theft.pkl', 'wb') as f:
  pickle.dump(results, f)


def plot_sarima_forecast(model_file, data,no_of_months):

  with open(model_file, 'rb') as f:
    model = pickle.load(f)


  forecast_horizon = no_of_months
  forecast = model.forecast(steps=forecast_horizon)


  fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Original Data", "Predicted Future Values"), vertical_spacing=0.3)


  fig.add_trace(go.Scatter(x=data.index, y=data["count"], name="Original Data"), row=1, col=1)


  fig.add_trace(go.Scatter(x=pd.date_range(start=data.index[-1], periods=forecast_horizon + 1, freq="M"), y=forecast, name="Predicted Future Values"), row=2, col=1)


  fig.update_layout(height=900, title_text="SARIMA Model for Theft Data")
  fig.show()


plot_sarima_forecast('crimeDetection\ML_models\sarima_model_theft.pkl', df_theft,36)

"""Create model for forecasting:Missing person Crime"""

# Define the SARIMA model
model = SARIMAX(df_missing_person["count"], order=(2, 1, 2), seasonal_order=(1, 0, 1, 12))


results = model.fit()

# Create a plotly subplot
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Original Data", "Predicted Future Values"), vertical_spacing=0.3)

# Plot the original data
fig.add_trace(go.Scatter(x=df_missing_person.index, y=df_missing_person["count"], name="Original Data"), row=1, col=1)

forecast_horizon = 12
forecast = results.forecast(steps=forecast_horizon)

# Plot the predicted future values
fig.add_trace(go.Scatter(x=pd.date_range(start=df_missing_person.index[-1], periods=forecast_horizon + 1, freq="M"), y=forecast, name="Predicted Future Values"), row=2, col=1)

# Update the layout and show the plot
fig.update_layout(height=900, title_text="SARIMA Model for Theft Data")
fig.show()

with open('crimeDetection\ML_models\sarima_model_missing_person.pkl', 'wb') as f:
  pickle.dump(results, f)



plot_sarima_forecast('crimeDetection\ML_models\sarima_model_theft.pkl', df_missing_person,12)

"""Sarima model to forecast Motor Vechicle Accidents Non Fatal Crime"""

# Define the SARIMA model
model = SARIMAX(df_motor_vehicle_accidents_non_fatal["count"], order=(2, 1, 2), seasonal_order=(1, 0, 1, 12))


results = model.fit()

# Create a plotly subplot
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Original Data", "Predicted Future Values"), vertical_spacing=0.3)

# Plot the original data
fig.add_trace(go.Scatter(x=df_motor_vehicle_accidents_non_fatal.index, y=df_motor_vehicle_accidents_non_fatal["count"], name="Original Data"), row=1, col=1)

forecast_horizon = 12
forecast = results.forecast(steps=forecast_horizon)

# Plot the predicted future values
fig.add_trace(go.Scatter(x=pd.date_range(start=df_motor_vehicle_accidents_non_fatal.index[-1], periods=forecast_horizon + 1, freq="M"), y=forecast, name="Predicted Future Values"), row=2, col=1)

# Update the layout and show the plot
fig.update_layout(height=900, title_text="SARIMA Model for Theft Data")
fig.show()

with open('crimeDetection\ML_models\sarima_model_vehicle_accidents_non_fatal.pkl', 'wb') as f:
  pickle.dump(results, f)

plot_sarima_forecast('crimeDetection\ML_models\sarima_model_vehicle_accidents_non_fatal.pkl', df_motor_vehicle_accidents_non_fatal,12)

