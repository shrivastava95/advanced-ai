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



# Define the time series
ARIMA_Sample = [6.988045118523939, 16.936183972662768, 27.894793932851556, 11.360123882026187, 
                6.796995800146647, 18.30869955623013, 34.51655112738614, 17.427290748999486, 
                14.758830109387528, 17.45345650377549, 34.32732873976919, 18.836214738732913, 
                11.221346213169166, 14.796636160332834, 27.559055952243373, 15.298984194488494, 
                13.564500020803997, 15.854546559775184, 25.6623452565195, 10.116605931110058]

d = 2
# difference the time series
ARIMA_Sample_original = ARIMA_Sample.copy()
ARIMA_Sample, residuals = difference(ARIMA_Sample, order=d)





# Estimate the ACF and PACF
lag = len(ARIMA_Sample) // 2 - 1
acf_vals = acf(ARIMA_Sample, nlags=lag)
pacf_vals = pacf(ARIMA_Sample, nlags=lag, method='ywm')
# plt.plot(acf_vals)
# plt.show()
# plt.plot(pacf_vals)
# plt.show()

# Determine the order of the AR model
p = 2

# Use the Yule-Walker equations to estimate the coefficients
r = acf_vals[1:p+1]
R = toeplitz(acf_vals[:p])
a = np.linalg.solve(R, r)

# Print the AR coefficients
print(a)

# Determine the order of the MA model
q = 5

# Use the innovations algorithm to estimate the MA coefficients
innovations = np.zeros(len(ARIMA_Sample))
for i in range(p, len(ARIMA_Sample)):
    slice = ARIMA_Sample[i-1-len(ARIMA_Sample):i-p-1-len(ARIMA_Sample):-1]
    pred = np.sum(np.array(slice) * np.array(a))
    innovations[i] = ARIMA_Sample[i] - pred

r = acf(innovations, nlags=q, fft=False)
R = toeplitz(r[:q])
ma = np.linalg.solve(R, r[1:])
print('MA coefficients:', ma)

def forecast(series, num_preds, AR_coeffs, MA_coeffs, centre):
    # Initialize the predicted values list with the original time series
    outputs = series.copy()
    outputs2 = series.copy()

    # Loop over the number of predictions to make
    for i in range(num_preds):
        # Calculate the prediction using the AR and MA coefficients
        ar_term = np.sum(np.array(outputs[-1:-(1+len(AR_coeffs)):-1]) * np.array(AR_coeffs))
        ma_term = np.sum(np.array(outputs[-1:-1-len(MA_coeffs):-1]) * np.array(MA_coeffs))
        pred = ar_term + ma_term

        # Add the predicted value to the output list
        outputs = np.append(outputs, pred + centre)
        outputs2 = np.append(outputs2, ar_term + centre)

    # Plot the original time series and the predicted values
    plt.title(f'AR({p}) model with D={0} order differencing')
    print(outputs, len(outputs))
    # outputs  = list(series) +  list(de_difference(outputs [len(series):],  residuals=residuals))
    # outputs2 = list(series) +  list(de_difference(outputs2[len(series):],  residuals=residuals))

    # outputs = de_difference(outputs, residuals)
    # outputs2 = de_difference(outputs2, residuals)
    # series = de_difference(series, residuals)
    plt.plot(outputs, label=f'MA term')
    plt.plot(outputs2, label=f'no MA term')
    plt.plot(series, label=f'original series')
    plt.legend()
    plt.show()


    outputs = de_difference(outputs, residuals)
    outputs2 = de_difference(outputs2, residuals)
    series = de_difference(series, residuals)
    plt.plot(outputs, label=f'MA term')
    plt.plot(outputs2, label=f'no MA term')
    plt.plot(series, label=f'original series')
    plt.legend()
    plt.show()

forecast(ARIMA_Sample, 15, a, ma, np.mean(ARIMA_Sample))

