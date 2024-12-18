#libraries for the machine learning models
import pmdarima as pm
#Libraries to prepare data
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
import sqlite3
from app.data import api_data, SqlRepository
load_dotenv()
#library to determine number of differencing to use
from app.decomposition import decompose
#libraries to save and load model
import joblib
from glob import glob
from pathlib import Path

class Arima:
    def __init__(self, ticker):
        self.ticker = ticker
        #instantiate name for the filepath for price model sub directory
        self.model_directory = os.environ.get('Model_directory')
        self.model1_subdirectory = os.environ.get('model1_subdirectory')
        
    #general function to get data
    def _wrangle(self, ticker, use_new_data=True):
        #setup connection to database
        connection = sqlite3.connect(database= os.environ.get('DB_NAME'), check_same_thread= False)
        #instantiate sql repo
        repo = SqlRepository(connection= connection)
        cursor = connection.cursor()
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
        #sort values in ascending order
        df.sort_values(by= 'Date', inplace= True)
        return df
    
    #function to get d
    def _get_d(self, series):
        sig_value= 0.05
        data= series
        model= decompose()
        difference_test= model.adf(data)
        if difference_test[1] < sig_value:
            return 0
        count= 1
        while difference_test[1] > sig_value:
            data= data.diff().dropna()
            difference_test= model.adf(data)
            count = +1
        return count
    #create model and fit data
    def fit_arima(self):
        data= self._wrangle(self.ticker, use_new_data=True)
        #get d
        d= self._get_d(data.Close)
        self.model= pm.auto_arima(data['Close'],
                    start_p=1, start_q=1,max_p=10, max_q=10,  
                       # frequency of series set to annual
                      d=d, # 'd' determined manually using the adf test
                      seasonal=False,  
                      start_P=1, start_Q=1, D=0,
                      trace=True,
                      error_action='ignore',  
                      suppress_warnings=True,
                      stepwise=False,
                     approximaton=False,
                     n_jobs= -1)
       #make forecast with the model
    def make_forecast(self, horizon):
        forecasts = self.model.predict(n_periods=horizon, alpha=0.05)
        forecasts_dict = forecasts.to_dict()
        return forecasts_dict
    #save model
    def dump(self):
       #create file path to save and store the price model
        filepath = os.path.join(self.model_directory, self.model1_subdirectory,(f'{self.ticker}.pkl'))
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        #save model
        joblib.dump(self.model, filepath)
        return filepath
        
    #load model
    def load(self):
        #prepare a pattern for glob search
        pattern = os.path.join(self.model_directory, self.model1_subdirectory, (f'*{self.ticker}.pkl'))
        try:
            model_path = sorted(glob(pattern))[-1]
        except IndexError:
            raise Exception(f"Oops No model trained for {self.ticker} chai..")
        self.model = joblib.load(model_path)
        return self.model
        
        
                
     