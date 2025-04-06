import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class Optimizer():
    def __init__(self):
        pass

def generateTestSamples(Nsamples = 5000):
    np.random.seed(42)
    X = 7.5 * np.random.rand(Nsamples, 1) + 0.5
    m, b = 4.1, 1.5
    noise = np.random.normal(loc=0.0, scale=0.6, size=(Nsamples, 1))
    y = m * X + b + (X ** 1.5) * noise
    return X, y, m, b

def predictWithRefModel(X, y):
    refModel = LinearRegression()
    refModel.fit(X.reshape(-1, 1), y)
    yPredRef = refModel.predict(X.reshape(-1, 1))
    mRef = refModel.coef_[0]
    bRef = refModel.intercept_
    return X, yPredRef, mRef, bRef

def main():
    # Create data points
    X, y, mTrue, bTrue = generateTestSamples(5000)
    print(f"True values: m={round(mTrue, 1)}, b={round(bTrue, 1)}")

    # Create reference model
    _, yPredRef, mRef, bRef = predictWithRefModel(X, y)
    print(f"Ref  values: m={round(float(mRef), 1)}, b={round(float(bRef), 1)}")

    # Chose optimizer
    currentOptimizer = "Gradient Descent"


    # Output
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, alpha=0.6, s=10, label="Data")
    plt.plot(X, yPredRef, color="red", label="Optimal Line", linewidth=2)
    plt.title(f"Linear Regression with {currentOptimizer}")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    main()
# np.random.seed(42)
#     X = 2.5 * np.random.rand(1000, 1) + 0.5
#     y = 1.0 + 0.8 * np.random.rand(1000, 1) * 0.3
#
# generate datapoints where I can find a lineare regression line.