from fastapi import FastAPI
from pydantic import BaseModel
import os
from dotenv import load_dotenv
import sqlite3
import requests
from app.model1 import Arima

#instantiate faspi app
app= FastAPI()

#get method for the '/hello' path with 200 status code
@app.get("/hello", status_code=200)
def hello():
    return{"message":"it works.."} 

#create 'FitIn' class
class FitIn(BaseModel):
    ticker: str
#create 'FitOut' class
class FitOut(FitIn):
    success: bool
    message: str
        
#post method for '/fit' path with 200 status code
@app.post("/fit", status_code=200, response_model=FitOut)
def fit_model(request:FitIn):
    #create 'response' dictionary from 'request'
    response= request.dict()
    #create try block to handle exceptions
    try:
        #instanstiate model
        model= Arima(ticker=request.ticker)
        #fit model
        model.fit_arima()
        #save the model
        filename= model.dump()
        #add success key to response
        response["success"]= True
        #add message key to respose
        response["message"]= f"model built and trained for '{filename}'."
    except Exception as e:
        #succes key to response
        response["success"]= False
        #message key to response
        response["message"]= str(e)
    return response

#create forecast class
class ForecastIn(BaseModel):
    ticker: str
    horizon: int
class ForecastOut(ForecastIn):
    success: bool
    forecast: dict
    message: str

#post method for '/Forecast' class
@app.post('/forecast', status_code=200, response_model=ForecastOut)
def make_forecast(request:ForecastIn):
    #create a response dictionary from request
    response= request.dict()
    #create a try block
    try:
        #instantiate model
        model_class= Arima(request.ticker)
        #load the stored model
        model_class.load()
        #generate forecasts
        forecasts= model_class.make_forecast(request.horizon)
        #add success key to response
        response["success"]= True
        #add forecast key
        response["forecast"]= forecasts
        #add message key
        response["message"]= f"Next {request.horizon} days Forecasted closing prices for BTC-USD"
    #create an exception
    except Exception as e:
        response["success"]= False
        response["forecast"]= {}
        response["message"]= str(e)
    return response
        
        
    