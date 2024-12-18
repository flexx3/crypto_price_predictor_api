#Librarie to prepare data
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import sqlite3
from app.data import api_data, SqlRepository
load_dotenv()

#library for decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
#library for running adf and kpss tests for stationarity
from statsmodels.tsa.stattools import adfuller, kpss

#import libraries for the visualization
from plotly.subplots import make_subplots
from plotly import graph_objects as go

class decompose:
    #general function to get data
    def wrangle(self, ticker, start_date, end_date, use_new_data=True):
        #setup connection to database
        connection = sqlite3.connect(database= os.environ.get('DB_NAME'), check_same_thread= False)
        #instantiate sql repo
        repo = SqlRepository(connection= connection)
        if use_new_data==True:
            #instantiate api_data
            api = api_data()
            records = api.get_data(ticker)
            #format the columns
            if records.columns.to_list() != ['Close', 'High', 'Low', 'Open', 'Volume']:
                column_list = records.columns.to_list()
                columns = [val[0] for val in column_list]
                records.columns= columns
            query = f"Drop Table If Exists '{ticker}' "
            cursor.execute(query)
            connection.commit()
            data = repo.insert_data(table_name= ticker, records= records, if_exists= 'replace')
        df = repo.read_table(ticker)
        connection.close()
        df = df.loc[start_date:end_date]
        df['Returns'] = np.log(df['Close']/df['Close'].shift(1))
        df['Volatility'] = df['Returns'].rolling(5).std()
        #sort values in ascending order
        df.sort_values(by= 'Date', inplace= True)
        #fill values
        df.fillna(method= 'ffill', inplace= True)
        if df.shape[0]== 0:
                raise Exception(f"""oops! wrong date range, data only available between 
                                {df.index[0].strftime('%Y-%m-%d')} and {df.index[-1].strftime('%Y-%m-%d')}""")
        return df
    #function for stochastic oscillator
    def get_stochastics_data(self, data, window=14):
            max_high = data['High'].rolling(window=window).max()
            min_low = data['Low'].rolling(window=window).min()
            fast_k = 100 * ((data['Close'] - min_low) / (max_high - min_low))
            slow_k = fast_k.rolling(window=3).mean()

            return fast_k, slow_k
    #function to determine model type for time series decomposition
    def model_threshold(self, data, window):
        #calculate standard deviation
        model_type= None
        rolling_std= data.rolling(window=window).std()
        #calculate rolling mean
        rolling_mean= data.rolling(window=window).mean()
        #calculate coefficient of variation(cv)
        cv= rolling_std / rolling_mean
        #calculate variance of the cv
        variance= cv.var()
        #set threshold value
        threshold= 0.02
        #identify model based on threshold and variance
        if variance < threshold:
            model_type= 'additive'
            return model_type
        else:
            model_type= 'multiplicative'
            return model_type
        
    #function to plot returns
    def plot_return(self, ticker, start_date, end_date):
        data = self.wrangle(ticker=ticker, start_date=start_date, end_date=end_date)
        data['26D EMA Returns'] = data['Returns'].ewm(span=26, adjust=False).mean()
        data['20D SMA Returns'] = data['Returns'].rolling(20).mean()
        data['%K'], data['%D']= self.get_stochastics_data(data)
        #prepare plots
        fig= make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.01, x_title='Date',row_heights=[800,600,600])
        fig.add_traces(
            [
                go.Scatter(x=data.index, y=data['Returns'], name='Returns'),
                go.Scatter(x=data.index, y=data['26D EMA Returns'], name='26D EMA Returns', line=dict(dash='dot')),
                go.Scatter(x=data.index, y=data['20D SMA Returns'], name='20D SMA Returns', line=dict(dash='dashdot', color='black')),
                go.Bar(x= data.index, y= data['Volume'], name= 'Trading Volume', marker=dict(color='blue')),
                go.Scatter(x= data.index, y= data['%K'], name= 'FasterStochasticOscillator(%K)'),
                go.Scatter(x= data.index, y= data['%D'], name= 'SlowerStochasticOscillator(%D)')
            ],
            rows=[1,1,1,2,3,3], cols=[1,1,1,1,1,1]
        )
        fig.update_layout(
            title= f' 26D and 20D EMA returns for {ticker}',
            yaxis_title= 'Returns',
            yaxis2_title= 'Volume',
            yaxis3_title= 'Stochastic Oscillator',
            xaxis_rangeslider_visible=False,
            width= 1000
        )
        return fig
    
    #function to visualize volatility
    def plot_volatility(self, ticker, start_date, end_date):
        data = self.wrangle(ticker=ticker, start_date=start_date, end_date=end_date)
        data['%K'], data['%D']= self.get_stochastics_data(data)
        #prepare plots
        fig= make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.01, x_title='Date',row_heights=[800,600,600])
        fig.add_traces(
            [
                go.Scatter(x=data.index, y=data['Volatility'], name='Volatility'),
                go.Bar(x= data.index, y= data['Volume'], name= 'Trading Volume', marker=dict(color='blue')),
                go.Scatter(x= data.index, y= data['%K'], name= 'FasterStochasticOscillator(%K)'),
                go.Scatter(x= data.index, y= data['%D'], name= 'SlowerStochasticOscillator(%D)')
            ],
            rows=[1,2,3,3], cols=[1,1,1,1]
        )
        fig.update_layout(
            title= f' Volatility for {ticker}',
            yaxis_title= 'Volatility',
            yaxis2_title= 'Volume',
            yaxis3_title= 'Stochastic Oscillator',
            xaxis_rangeslider_visible=False,
            width= 1000
        )
        return fig
    #decompose time series volatility
    def decompose_volatility(self, ticker, start_date, end_date):
        #get the data
        data= self.wrangle(ticker=ticker, start_date=start_date, end_date=end_date)
        #drop missing values
        data.dropna(inplace=True)
        #determine model type
        model_type= self.model_threshold(data['Volatility'], window=5)
        #decompose data
        decomposition= seasonal_decompose(data['Volatility'], model= model_type)
        #extract the decomposed components
        observed= decomposition.observed
        trend= decomposition.trend
        seasonal= decomposition.seasonal
        residual= decomposition.resid
        fig=  make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05, x_title='Date',row_heights=[600,600,600,600])
        fig.add_traces(
            [
                go.Scatter(x=observed.index, y=observed, mode='lines', name='Observed'),
                go.Scatter(x=trend.index, y=trend, mode='lines', name='Trend', line=dict(dash='dot', color='blue')),
                go.Scatter(x=seasonal.index, y=seasonal, mode='lines', name='Seasonal', line=dict(dash='dash', color='green')),
                go.Scatter(x=residual.index, y=residual, mode='lines', name='Residual', line=dict(dash='dashdot', color='red'))
            ],
            rows= [1,2,3,4], cols= [1,1,1,1]
        )
        fig.update_layout(
            title= f'Volatility Time Series Decomposition For {ticker} Weekly Trading Cycle',
            yaxis_title= 'Observed',
            yaxis2_title= 'Trend',
            yaxis3_title= 'Seasonality',
            yaxis4_title= 'Residuals',
            xaxis_rangeslider_visible=False,
            width= 1000
        )
        return fig
    #decompose time series closing price
    def decompose_price(self, ticker, start_date, end_date):
        #get the data
        data= self.wrangle(ticker=ticker, start_date=start_date, end_date=end_date)
        data['Close']= data['Close'].rolling(21).mean()
        #drop nan
        data.dropna(inplace= True)
        #determine model type
        model_type= self.model_threshold(data['Volatility'], window=21)
        #decompose data
        decomposition= seasonal_decompose(data['Close'], model= model_type)
        #extract the decomposed components
        observed= decomposition.observed
        trend= decomposition.trend
        seasonal= decomposition.seasonal
        residual= decomposition.resid
        fig=  make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05, x_title='Date',row_heights=[600,600,600,600])
        fig.add_traces(
            [
                go.Scatter(x=observed.index, y=observed, mode='lines', name='Observed'),
                go.Scatter(x=trend.index, y=trend, mode='lines', name='Trend', line=dict(dash='dot', color='blue')),
                go.Scatter(x=seasonal.index, y=seasonal, mode='lines', name='Seasonal', line=dict(dash='dash', color='green')),
                go.Scatter(x=residual.index, y=residual, mode='lines', name='Residual', line=dict(dash='dashdot', color='red'))
            ],
            rows= [1,2,3,4], cols= [1,1,1,1]
        )
        fig.update_layout(
            title= f'Time Series Decomposition For Closing Price of {ticker} For Monthly Trading Cycle',
            yaxis_title= 'Observed',
            yaxis2_title= 'Trend',
            yaxis3_title= 'Seasonality',
            yaxis4_title= 'Residuals',
            xaxis_rangeslider_visible=False,
            width= 1000
        )
        return fig
    
    #perform augmented dickey-fuller test for non-stationarity
    def adf(self, series):
        '''
        Null Hypotesis: Data is not stationary
        Alternate Hypothesis: Data is stationary
        
        '''
        indices= ['Test Statistic', 'p-value', 'No of Lags', 'No of Observations']
        adf_test= adfuller(series, autolag= 'AIC')
        result= pd.Series(adf_test[0:4], index= indices)
        for key, value in adf_test[4].items():
            result[f'Critical_value {key}']= value
        return result
    #perform kpss test for stationarity
    def kpss_test(self,series, h0_type= 'c'):
        '''
        Null Hypotesis: Data is stationary
        Alternate Hypothesis: Data is not stationary
        
        '''
        indices= ['Test Statistic', 'p-value', 'No of Lags']
        kpss_test= kpss(series, regression= h0_type)
        result= pd.Series(kpss_test[0:3], index= indices)
        for key, values in kpss_test[3].items():
            result[f'Critical_value {key}']= value
        return result
    
    