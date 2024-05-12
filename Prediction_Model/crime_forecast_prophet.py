# -*- coding: utf-8 -*-
"""Crime_Forecast_Prophet.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1uy7naZimhK_LjWolxv578n5Zx2UQUxyY
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import  seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import json
from prophet.serialize import model_from_json
with open('/content/prophet_model (1).json', 'r') as fin:
    m1 = model_from_json(json.load(fin))  # Load model

import plotly.express as px
def forecast_prophet_plot(x):
  pred = m1.make_future_dataframe(periods=x ,freq = "M")
  forecast = m1.predict(pred)
  figure = m1.plot(forecast, xlabel='Date', ylabel='Crime Rate')
  px.line(forecast, x='ds', y='yhat', title='Forecasted Crime Rate')
  return figure

import pandas as pd
df = pd.read_csv("/content/prophet_data.csv", index_col=0)

df.head()

# prompt: in the above function plot the original data points using plotly

def forecast_prophet_plot(x):
  pred = m1.make_future_dataframe(periods=x ,freq = "M")
  forecast = m1.predict(pred)
  figure = m1.plot(forecast, xlabel='Date', ylabel='Crime Rate')
  px.line(forecast, x='ds', y='yhat', title='Forecasted Crime Rate')
  px.scatter(df, x="ds", y="y", title="Original Crime Rate Data")  # Plot original data points
  return figure

forecast_prophet_plot(34)