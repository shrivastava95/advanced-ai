

import numpy as np
from matplotlib import pyplot as plt

def difference(series, order):
    residuals = []
    for i in range(order):
        residuals.append(series[0])
        series = np.diff(series)
    return series, residuals

def de_difference(series, residuals):
    residuals = residuals[::]
    series = series[::]
    print(series)
    while residuals:
        series[0] += residuals[-1]
        for i in range(1, len(series)):
            series[i] += series[i-1]
        series = [residuals.pop()] + list(series)
    return series


series = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

differenced, residual = difference(series, 2)
print(residual)
de_differenced = de_difference(differenced, residual)
print(differenced)
print(de_differenced)