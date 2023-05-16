import numpy as np
import pmdarima as pm
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm

def calculate_ar_coefficients_yw(time_series, p):
    """
    Calculates the AR coefficients of a time series using the Yule-Walker method.
    
    Args:
    time_series (list): The time series data.
    p (int): The number of AR coefficients to estimate.
    
    Returns:
    list: The estimated AR coefficients.
    """
    
    n = len(time_series)
    
    # calculate the autocorrelation function (ACF) of the time series
    acf = np.zeros(p+1)
    for k in range(p+1):
        acf[k] = np.sum((time_series[k:n] - np.mean(time_series)) * 
                        (time_series[0:n-k] - np.mean(time_series))) / (n-1)
    
    # create the Yule-Walker matrix
    yw_matrix = np.zeros((p,p))
    for i in range(p):
        for j in range(p):
            yw_matrix[i,j] = acf[abs(i-j)]
    
    # solve the Yule-Walker equations to get the AR coefficients
    ar_coef = np.zeros(p)
    yw_vector = acf[1:p+1]
    ar_coef = np.linalg.solve(yw_matrix, yw_vector)
    
    # # scale the coefficients using the variance of the time series

    # var = np.var(time_series)
    # ar_coef = ar_coef * np.sqrt(var)

    return ar_coef


def ma_innovations(ts, q):
    """
    Estimates the q MA coefficients of a time series using the innovations algorithm.

    Parameters:
        - ts (list or numpy array): the time series
        - q (int): the number of MA coefficients to estimate

    Returns:
        - ma_coefs (numpy array): an array of shape (q,) containing the estimated MA coefficients
    """
    n = len(ts)
    e = ts.copy()
    ts=np.array(ts)
    e=np.array(e)
    ma_coefs = np.zeros(q)
    
    for i in range(q):
        ma_coefs[i] = np.sum(e[i+1:] * ts[:n-i-1]) / np.sum(ts[:n-i-1]**2)
        e[i+1:] -= ma_coefs[i] * ts[:n-i-1]

    return ma_coefs




def get_moving_average_coeffs(time_series, Q):
    """
    Calculates the moving average coefficients for a time series using least squares estimation.

    Parameters:
    -----------
    time_series : pd.Series
        A pandas series object containing the time series data.
    Q : int
        The window size for the moving average.

    Returns:
    --------
    coefficients : np.array
        A numpy array containing the moving average coefficients.
    """

    # Create a matrix of lagged values for the moving average model
    X = np.zeros((len(time_series) - Q, Q))
    for i in range(Q):
        X[:, i] = time_series[i:len(time_series) - Q + i]

    # Create the dependent variable (i.e., the observed data)
    y = time_series[Q:]

    # Fit the linear regression model using least squares estimation
    coefficients, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    # Return the estimated coefficients
    return coefficients






def ARIMA_Forecast(input_series:list, P: int, D: int, Q: int, prediction_count: int)->list:
    # Complete the ARIMA model for given value of input series, P, Q and D
    # return n predictions after the last element in input_series
    # create the differenced series
    # Apply differencing
    diff_data = input_series
    for i in range(D):
        diff_data = [diff_data[j+1] - diff_data[j] for j in range(len(diff_data)-1)]

    # Initialize forecast with last available data point
    ar_coef=calculate_ar_coefficients_yw(input_series, P)
    print(ar_coef)
    # ma_coef=ma_innovations(input_series,Q)
    # print(ma_coef)
    ma_coef = get_moving_average_coeffs(input_series, Q)
    print(ma_coef)
    forecast = [diff_data[-1]]
    
    # ar_coef=[0.1]*P
    # ma_coef=[0.1]*Q
    # Loop over forecast steps
    for i in range(prediction_count):
        # Calculate the next forecast value
        next_value = 0
        for j in range(len(ar_coef)):
            if len(forecast) - j - 1 < 0:
                continue
            next_value += ar_coef[j] * forecast[-j-1]
        for j in range(len(ma_coef)):
            if len(diff_data) - j - 1 < 0:
                continue
            next_value += ma_coef[j] * diff_data[-j-1]
        forecast.append(next_value)

    # Reverse differencing
    for i in range(D):
        for j in range(len(forecast)-1, 0, -1):
            forecast[j] += forecast[j-1]
        forecast.pop(0)

    return forecast
    # return [0] * prediction_count 
    # This is wrong, but it does create a list of the required size, 
    # and is used to showcase the `Plot()` function in the test program 


def HoltWinter_Forecast(input:list, alpha:float, beta:float, gamma:float, seasonality:int, prediction_count: int)->list:
    # Complete the Holt-Winter's model for given value of input series
    # Use either of Multiplicative or Additve model
    # return n predictions after the last element in input_series
    """
    Holt-Winters forecast implementation for seasonal time series data.

    Parameters:
    -----------
    input_series : list or numpy array
        A list or numpy array containing the input time series data.
    alpha : float
        Smoothing parameter for level.
    beta : float
        Smoothing parameter for trend.
    gamma : float
        Smoothing parameter for seasonality.
    seasonality : int
        The length of the seasonal period.
    prediction_count : int
        The number of time steps to forecast after the input series.
    additive : bool, optional (default=True)
        Whether to use the additive or multiplicative model.

    Returns:
    --------
    forecast : list
        A list containing the predicted values for the time series.
    """

    # Initialize forecast and parameter lists
    forecast = []
    print(seasonality, 'seasonality')
    l = [np.nan] * seasonality
    b = [np.nan] * seasonality
    s = [np.mean(input[i::seasonality]) for i in range(seasonality)]

    # Choose model type based on input
    # if additive:
    def forecast_func(alpha, beta, gamma, l, b, s, h):
        return l[-1] + h * b[-1] + s[-(seasonality - h)]
    # else:
    #     def forecast_func(alpha, beta, gamma, l, b, s, h):
    #         return (l[-1] + h * b[-1]) * s[-(seasonality - h)]

    # Iterate through input series and make forecasts
    for i, x in enumerate(input):

        # Seasonal indices for this point
        t = i % seasonality

        # Initialize level and trend components
        if i < seasonality:
            l[t] = x
            b[t] = np.mean([input[seasonality + t] - input[t] for t in range(seasonality)]) / seasonality

        # Holt-Winters algorithm
        if not np.isnan(x):
            # if additive:
                l_prev, l[t] = l[t], alpha * (x - s[-(seasonality - t)]) + (1 - alpha) * (l[t] + b[t])
                b[t] = beta * (l[t] - l_prev) + (1 - beta) * b[t]
                s[t] = gamma * (x - l[t]) + (1 - gamma) * s[-(seasonality - t)]
            # else:
            #     l_prev, l[t] = l[t], alpha * (x / s[-(seasonality - t)]) + (1 - alpha) * (l[t] + b[t])
            #     b[t] = beta * (l[t] - l_prev)+ + (1 - beta) * b[t]
            #     s[t] = gamma * (x / (l_prev + b[t])) + (1 - gamma) * s[-(seasonality - t)]

        # Forecast next point in series
        forecast.append(forecast_func(alpha, beta, gamma, l, b, s, (i + prediction_count) % seasonality))

    return forecast

    # return [0] * prediction_count 
    # This is wrong, but it does create a list of the required size, 
    # and is used to showcase the `Plot()` function in the test program 


def ARIMA_Paramters(input_series:list)->tuple: # (P, D, Q)
    arima_model=pm.auto_arima(input_series,trace=True,error_action='ignore',suppress_warnings=True)
    P,D,Q=arima_model.order
    print(P,' ',D,' ',Q)
    return P,D,Q
    # pass

def HoltWinter_Parameters(input_series:list)->tuple: # (Alpha, Beta, Gamma, Seasonality)
    # pass
    # Initialize parameters
    min_aic = np.inf
    best_params = None
    
    # Try different combinations of parameters and choose the one with the lowest AIC
    for a in np.arange(0.1, 1.0, 0.1):
        for b in np.arange(0.1, 1.0, 0.1):
            for g in np.arange(0.1, 1.0, 0.1):
                for s in range(2, int(len(input_series)/2)):
                    try:
                        # Fit model with current set of parameters
                        model = ExponentialSmoothing(input_series, seasonal_periods=s, trend='add', seasonal='add').fit(smoothing_level=a, smoothing_slope=b, smoothing_seasonal=g)
                        
                        # Check if AIC of current model is lower than minimum AIC so far
                        if model.aic < min_aic:
                            min_aic = model.aic
                            best_params = (a, b, g, s)
                    except:
                        pass
    print(a,' ',b,' ',g,' ',s)
    return best_params