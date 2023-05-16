import numpy as np

def ARIMA_Forecast(input_series:list, P: int, D: int, Q: int, prediction_count: int)->list:
    # Complete the ARIMA model for given value of input series, P, Q and D
    # return n predictions after the last element in input_series
    
    # return [0] * prediction_count 
    # This is wrong, but it does create a list of the required size, 
    # and is used to showcase the `Plot()` function in the test program 


def HoltWinter_Forecast(input:list, alpha:float, beta:float, gamma:float, seasonality:int, prediction_count: int)->list:
    # Complete the Holt-Winter's model for given value of input series
    # Use either of Multiplicative or Additve model
    # return n predictions after the last element in input_series


    return [0] * prediction_count 
    # This is wrong, but it does create a list of the required size, 
    # and is used to showcase the `Plot()` function in the test program 


def ARIMA_Paramters(input_series:list)->tuple: # (P, D, Q)
    pass

def HoltWinter_Parameters(input_series:list)->tuple: # (Alpha, Beta, Gamma, Seasonality)
    pass