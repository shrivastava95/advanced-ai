import pmdarima
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as smapi
import matplotlib.pyplot as plt #
import numpy as np

import numpy as np
from statsmodels.tsa.stattools import acf, pacf
from scipy.linalg import toeplitz
from matplotlib import pyplot as plt



# define the differencing and the anti-differencing functions
def difference(series, order):
    residuals = []
    for i in range(order):
        residuals.append(series[0])
        series = np.diff(series, n=1)
    return series, residuals

def de_difference(series, residuals):
    residuals = residuals[::]
    series = series[::]
    while residuals:
        series[0] += residuals[-1]
        for i in range(1, len(series)):
            series[i] += series[i-1]
        series = [residuals.pop()] + list(series)
    return series




def ARIMA_Forecast(input_series:list, P: int, D: int, Q: int, prediction_count: int)->list:
    # Complete the ARIMA model for given value of input series, P, Q and D
    # return n predictions after the last element in input_series
    
    # difference the time series
    ARIMA_Sample = input_series
    ARIMA_Sample_original = ARIMA_Sample.copy()
    ARIMA_Sample, residuals = difference(ARIMA_Sample, order=D)

    # Estimate the ACF and PACF
    lag = len(ARIMA_Sample) // 2 - 1
    acf_vals = acf(ARIMA_Sample, nlags=lag)
    pacf_vals = pacf(ARIMA_Sample, nlags=lag, method='ywm')

    # Use the Yule-Walker equations to estimate the coefficients
    r = acf_vals[1:P+1]
    R = toeplitz(acf_vals[:P])
    a = np.linalg.solve(R, r)

    # Print the AR coefficients
    print(a)


        ###########################################
    # # Use the innovations algorithm to estimate the MA coefficients
    # innovations = np.zeros(len(ARIMA_Sample))
    # for i in range(P, len(ARIMA_Sample)):
    #     slice = ARIMA_Sample[i-1-len(ARIMA_Sample):i-P-1-len(ARIMA_Sample):-1]
    #     pred = np.sum(np.array(slice) * np.array(a))
    #     innovations[i] = ARIMA_Sample[i] - pred
    # r = acf(innovations, nlags=Q, fft=False)
    # R = toeplitz(r[:Q])
    # ma = np.linalg.solve(R, r[1:])
    # print('MA coefficients:', ma)


    # Create a matrix of lagged values for the moving average model
    X = np.zeros((len(ARIMA_Sample) - Q, Q))
    for i in range(Q):
        X[:, i] = ARIMA_Sample[i:len(ARIMA_Sample) - Q + i]

    # Create the dependent variable (i.e., the observed data)
    y = ARIMA_Sample[Q:]

    # Fit the linear regression model using least squares estimation
    ma, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    ###########################################


    def forecast(series, num_preds, AR_coeffs, MA_coeffs, centre):
        # Initialize the predicted values list with the original time series
        outputs  = series.copy()
        outputs2 = series.copy()

        # Loop over the number of predictions to make
        for i in range(num_preds):
            # Calculate the prediction using the AR and MA coefficients
            ar_term = np.sum(np.array(outputs[-1:-(1+len(AR_coeffs)):-1]) * np.array(AR_coeffs))
            ma_term = np.sum(np.array(outputs[-1:-1-len(MA_coeffs):-1]) * np.array(MA_coeffs))
            pred = ar_term + ma_term

            # Add the predicted value to the output list
            outputs = np.append(outputs, pred)
            outputs2 = np.append(outputs2, ar_term)

        # Plot the original time series and the predicted values
        plt.title(f'AR({P}) model with D={D} order differencing')
        print(outputs, len(outputs))
        # outputs  = list(series) +  list(de_difference(outputs [len(series):],  residuals=residuals))
        # outputs2 = list(series) +  list(de_difference(outputs2[len(series):],  residuals=residuals))


        # plt.plot(outputs, label=f'MA term')
        # plt.plot(outputs2, label=f'no MA term')
        # plt.plot(series, label=f'original series')
        # plt.legend()
        # # plt.show()


        outputs = de_difference(outputs, residuals)
        # outputs2 = de_difference(outputs2, residuals)
        # series = de_difference(series, residuals)


        # plt.plot(outputs, label=f'MA term')
        # plt.plot(outputs2, label=f'no MA term')
        # plt.plot(series, label=f'original series')
        # plt.legend()
        # # plt.show()

        return outputs[len(series):]

    outputs = forecast(ARIMA_Sample, 15, a, ma, np.mean(ARIMA_Sample))

    return outputs[1:]
    # return [0] * prediction_count 
    # This is wrong, but it does create a list of the required size, 
    # and is used to showcase the `Plot()` function in the test program 


def HoltWinter_Forecast(input:list, alpha:float, beta:float, gamma:float, seasonality:int, prediction_count:int):
    forecast = []
    print(seasonality)
    
    
    # Initialized trend and level to a good starting value in order to make the forecasting converge faster
    level = input[-seasonality:]
    trend = [(input[-seasonality:][i] - input[-2*seasonality:-seasonality][i]) / seasonality for i in range(seasonality)]
    seasonality_components = input[-seasonality:]   
    print('bruh')

    # Define the forecast function for neater inference
    def forecast_func(level, trend, seasonality_components, h):
        pretrend = trend[-1] if trend[-1] is not None else 0
        prelevel = level[-1] if level[-1] is not None else 0
        return prelevel + h * pretrend + seasonality_components[-(seasonality - h)]

    # Iterate through the input series and make forecasts
    for i, x in enumerate(input):

        # Apply the Holt-Winters algorithm to the series, updating the level, trend and seasonality values.
        if x is not None:
            level_prev, level[i % seasonality] = level[i % seasonality], alpha * (x - seasonality_components[-(seasonality - i % seasonality)]) + (1 - alpha) * (level[i % seasonality] + trend[i % seasonality])
            trend[i % seasonality] = beta * (level[i % seasonality] - level_prev) + (1 - beta) * trend[i % seasonality]
            seasonality_components[i % seasonality] = gamma * (x - level[i % seasonality]) + (1 - gamma) * seasonality_components[-(seasonality - i % seasonality)]

        forecast.append(forecast_func(level, trend, seasonality_components, (i + prediction_count) % seasonality))

    return forecast


def ARIMA_Paramters(input_series:list)->tuple: # (P, D, Q)
    arima_model = pmdarima.auto_arima(input_series,trace=True,error_action='ignore',suppress_warnings=True)
    P, D, Q = arima_model.order
    print(f'ARIMA parameters:      P={P}    D={D}    Q={Q}')

    return P, D, Q

def HoltWinter_Parameters(input_series:list)->tuple: # (Alpha, Beta, Gamma, Seasonality)

    best_aic = None
    range_list = [0.9, 0.8, 0.6, 0.4]
    for alpha in range_list:
        for beta in range_list:
            for gamma in range_list:
                for seasonality in range(2, len(input_series)//2):
                    current_aic = ExponentialSmoothing(endog=input_series, seasonal_periods=seasonality, trend='add', seasonal='add').fit(smoothing_level=alpha, smoothing_slope=beta, smoothing_seasonal=gamma).aic
                    if best_aic == None or best_aic > current_aic:
                        best_alpha = alpha
                        best_beta = beta
                        best_gamma = gamma
                        best_seasonality = seasonality
    print(f'HOLT-WINTER parameters:      alpha={best_alpha}    beta={best_beta}    gamma={best_gamma}    seasonality={best_seasonality}')
    return [best_alpha, best_beta, best_gamma, best_seasonality]
