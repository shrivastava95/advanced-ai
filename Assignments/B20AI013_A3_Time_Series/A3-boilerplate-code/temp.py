def differencing(n):
    calc = [1]
    for _ in range(n):
        temp1 = calc + [0]
        temp2 = [0] + calc
        calc = [temp1[i] - temp2[i] for i in range(len(temp1))]
    return calc

print([differencing(i) for i in range(5)])




import numpy as np
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Load the Iris dataset
iris = load_iris()

# Fit an LDA model to the data
model = LinearDiscriminantAnalysis()
model.fit(iris.data, iris.target)

# Get the coefficients of the linear discriminants
coeffs = model.scalings_

# Check if the coefficients are all nonzero
linearly_separable = np.all(coeffs != 0)
print("Linearly separable: {}".format(linearly_separable))