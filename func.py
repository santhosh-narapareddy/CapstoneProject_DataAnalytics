# Basic libraries
import streamlit as st
import pandas as pd
import numpy as np
import base64
from PIL import Image
from stocknews import StockNews
from math import sqrt
## DateTime libraries
# from datetime import date as dt
import datetime
import time
import yfinance as yf
import pandas_datareader as pdr
from statsmodels.tsa.seasonal import seasonal_decompose
## Keras and SKLearn libraries for LSTM model
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM,Dropout
from keras import optimizers
from keras.callbacks import Callback
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error, r2_score,mean_absolute_percentage_error
## For XGBoost model
import os
import xgboost as xgb
from xgboost import XGBRegressor
from finta import TA
# For FB-Prophet model
from prophet import Prophet
# Plotly libraries
import plotly.graph_objects as go
import plotly.express as px
# Create plot for actual and predicted values
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt


def get_macro_data(country):
    # Start and end date to download max available data
    start_date = datetime.datetime(1900, 1, 1)
    end_date = datetime.datetime.now()
    
    #plot function
    def plot_data(data, title,x,y):
        st.write(f'<div style="text-align: center;"><h3>{title}</h3></div>', unsafe_allow_html=True)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['value'], mode='lines', name=title))
        #    fig.update_layout(title=title)
        fig.update_layout(
            xaxis=dict(tickfont=dict(size=20)),
            yaxis=dict(tickfont=dict(size=20)),
            font=dict(size=16)
        )
        fig.update_layout(xaxis_title=x, yaxis_title=y)
        return fig
    
    if country == 'USA':
        		
        #USA Variables
        usa_gdp = 'GDPA'
        usa_inflation = 'CPIAUCSL'
        usa_unemployment = 'UNRATE'
        usa_bank_rates = 'FEDFUNDS'
        usa_bond_rates = 'IRLTLT01USM156N'
        
        #USA data retrieval 
        gdp = pdr.get_data_fred(usa_gdp, start_date, end_date) / 1000
        gdp.columns = ['value']
        inflation = pdr.get_data_fred(usa_inflation, start_date, end_date)
        inflation.columns = ['value']
        unemployment = pdr.get_data_fred(usa_unemployment, start_date, end_date)
        unemployment.columns = ['value']
        bank_rate = pdr.get_data_fred(usa_bank_rates, start_date, end_date)
        bank_rate.columns = ['value']
        bond_rate = pdr.get_data_fred(usa_bond_rates, start_date, end_date)
        bond_rate.columns = ['value']
        
        #USA plots
        st.plotly_chart(plot_data(gdp,title="Gross Domestic Product",x="Year",y="Trillions $"),use_container_width=True)
        st.write("---")
        st.plotly_chart(plot_data(inflation,title="Inflation",x="Year",y="Units"),use_container_width=True)
        st.write("---")
        st.plotly_chart(plot_data(unemployment,title="Unemployment Rate",x="Year",y="Rate %"),use_container_width=True)
        st.write("---")
        st.plotly_chart(plot_data(bank_rate,title="Bank Rates",x="Year",y="Rate %"),use_container_width=True)
        st.write("---")
        st.plotly_chart(plot_data(bond_rate,title="Bond Rates",x="Year",y="Rate %"),use_container_width=True)
        
    elif country == 'Canada':
        #Canada variables
        canada_gdp = 'NGDPRXDCCAA'
        canada_bond_rates = 'IRLTLT01CAM156N'
        canada_unemployment = 'LRUNTTTTCAM156S'
        canada_inflation = 'CPALCY01CAM661N'
        canada_bank_rates = 'IR3TIB01CAM156N'
        
        # Canada data retrieval
        gdp = pdr.get_data_fred(canada_gdp, start_date, end_date) / 1e6  # Convert to trillions
        gdp.columns = ['value']
        inflation = pdr.get_data_fred(canada_inflation, start_date, end_date)
        inflation.columns = ['value']
        unemployment = pdr.get_data_fred(canada_unemployment, start_date, end_date)
        unemployment.columns = ['value']
        bank_rate = pdr.get_data_fred(canada_bank_rates, start_date, end_date)
        bank_rate.columns = ['value']
        bond_rate = pdr.get_data_fred(canada_bond_rates, start_date, end_date)
        bond_rate.columns = ['value']
        
        #Canada Plots
        st.plotly_chart(plot_data(gdp,title="Gross Domestic Product",x="Year",y="Trillions $"),use_container_width=True)
        st.write("---")
        st.plotly_chart(plot_data(inflation,title="Inflation",x="Year",y="Units"),use_container_width=True)
        st.write("---")
        st.plotly_chart(plot_data(unemployment,title="Unemployment Rate",x="Year",y="Rate %"),use_container_width=True)
        st.write("---")
        st.plotly_chart(plot_data(bank_rate,title="Bank Rates",x="Year",y="Rate %"),use_container_width=True)
        st.write("---")
        st.plotly_chart(plot_data(bond_rate,title="Bond Rates",x="Year",y="Rate %"),use_container_width=True)
    else:
        #India variables
        india_gdp = 'MKTGDPINA646NWDB'
        india_bond_rates = 'INDIRLTLT01STM'
        india_inflation = 'INDCPIALLMINMEI'
        india_bank_rates = 'INDIR3TIB01STM'
        
        #India data retrieval
        gdp = pdr.get_data_fred(india_gdp, start_date, end_date) / 1e12  # Convert to trillions
        gdp.columns = ['value']
        inflation = pdr.get_data_fred(india_inflation, start_date, end_date)
        inflation.columns = ['value']
        unemployment = None  # Data not available for India
        bank_rate = pdr.get_data_fred(india_bank_rates, start_date, end_date)
        bank_rate.columns = ['value']
        bond_rate = pdr.get_data_fred(india_bond_rates, start_date, end_date)
        bond_rate.columns = ['value']
        
        #India Plots
        st.plotly_chart(plot_data(gdp,title="Gross Domestic Product",x="Year",y="Trillions $"),use_container_width=True)
        st.write("---")
        st.plotly_chart(plot_data(inflation,title="Inflation",x="Year",y="Units"),use_container_width=True)
        st.write("---")
        st.plotly_chart(plot_data(bank_rate,title="Bank Rates",x="Year",y="Rate %"),use_container_width=True)
        st.write("---")
        st.plotly_chart(plot_data(bond_rate,title="Bond Rates",x="Year",y="Rate %"),use_container_width=True)
    

def load_data(file):
	df = pd.read_csv(file)
	return df

def fetch_index_data(country):
    tickers = {
        'India': '^NSEI',
        'Canada': '^GSPTSE',
        'USA': '^IXIC ^GSPC ^DJI'
    }

    selected_tickers = tickers.get(country).split()

    data = yf.download(selected_tickers, start=datetime.date.today() - datetime.timedelta(days=365), end=datetime.date.today(), group_by='ticker')

    return data

def get_start_end_dates(selected_duration):
    end_date = datetime.date.today()
    if selected_duration == "5 Year":
        start_date = end_date - datetime.timedelta(days=5*365)
    elif selected_duration == "7 Years":
        start_date = end_date - datetime.timedelta(days=7*365)  # Approximation
    elif selected_duration == "10 Years":
        start_date = end_date - datetime.timedelta(days=10*365)  # Approximation
    else:
        start_date = None
    return start_date, end_date

def create_dataset(X, y, time_step):
    Xs, ys = [], []
    for i in range(len(X) - time_step):
        v = X[i:(i + time_step)]
        Xs.append(v)
        ys.append(y[i + time_step])
    return np.array(Xs), np.array(ys)

def color_coded_sentiment(score):
    if score > 0.5:
        color = 'green'
    elif score < -0.5:
        color = 'red'
    else:
        color = 'yellow'
    return f'<span style="color:{color};">{score:.4f}</span>'

def fetch_index_dataa(ticker, period):
    
    try:
        data = yf.download(ticker, period=period, interval='1mo')
    except Exception as e:
        data = e
    return data

def display_country_index(data, key, title):
    st.write(f'<div style="text-align: center;"><h3>{title}</h3></div>', unsafe_allow_html=True)
    # index_data = data['Adj Close'] if not additional_data else data[key]['Adj Close']
    index_data = data['Adj Close']
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=index_data.index, y=index_data.values, mode='lines', name=key))
    fig.update_layout(
        xaxis=dict(tickfont=dict(size=20)),
        yaxis=dict(tickfont=dict(size=20)),
        font=dict(size=16),
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Adding decomposition graph
    decomposition = seasonal_decompose(index_data.dropna(), model='additive', period=12)
    fig_dec = go.Figure()
    fig_dec.add_trace(go.Scatter(x=index_data.index, y=decomposition.trend, mode='lines', name='Trend'))
    fig_dec.add_trace(go.Scatter(x=index_data.index, y=decomposition.seasonal, mode='lines', name='Seasonal'))
    fig_dec.add_trace(go.Scatter(x=index_data.index, y=decomposition.resid, mode='lines', name='Residual'))
    fig_dec.update_layout(
        xaxis=dict(tickfont=dict(size=20)),
        yaxis=dict(tickfont=dict(size=20)),
        font=dict(size=16),
        title='Time Series Decomposition'
    )
    st.plotly_chart(fig_dec, use_container_width=True)

    st.markdown("---")

