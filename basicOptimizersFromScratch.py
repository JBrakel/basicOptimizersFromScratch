import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

class Optimizer():
    def __init__(self, currentOptimizer, X, y, learningRate=0.01, epochs=50):
        self.currentOptimizer = currentOptimizer
        self.X, self.y = X, y
        self.learningRate = learningRate
        self.epochs = epochs
        self.mInit, self.bInit = None, None
        self.mPred, self.bPred = None, None
        self.historyParams, self.mse = [], []

    def initWeights(self):
        # Random initialization for m (slope) and b (intercept)
        mInit = np.random.randn() * 0.01
        bInit = np.random.randn() * 0.01
        wInit = mInit, bInit
        return wInit

    def saveLineParamsAndErrors(self, w, error):
        self.historyParams.append(w)
        self.mse.append(error)

    def getLineParamsAndErrors(self):
        return self.historyParams, self.mse

    def computeGradient(self, w, N):
        m, b = w
        yPred = self.X * m + b
        error = yPred - self.y

        # Gradient of MSE with respect to w and b
        dm = (2 / N) * np.dot(self.X.flatten(), error.flatten())
        db = (2 / N) * np.sum(error)
        dw = np.array([dm, db])

        # Store values to plot
        self.saveLineParamsAndErrors(w, np.mean(np.square(error)))
        return dw


    def gradientDescent(self):
        w = self.initWeights()
        Nsamples = len(self.X)
        for _ in range(self.epochs):
            dw = self.computeGradient(w, Nsamples)
            w = w - self.learningRate * dw

        self.mPred, self.bPred = w
        yPred = w[0] * self.X + w[1]
        return self.X, yPred, w[0], w[1]



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

    # Choose optimizer
    currentOptimizer = "Gradient Descent"
    opt = Optimizer(currentOptimizer,
                    X, y,
                    learningRate=0.001,
                    epochs=200)

    _, yPred, mPred, bPred = opt.gradientDescent()
    historyParams, mse = opt.getLineParamsAndErrors()
    print(f"Pred values: m={round(float(mPred), 1)}, b={round(float(bPred), 1)}")

    # Output
    # Create a figure with two subplots: one for the regression lines and one for the error
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    # Plot the regression lines on the first subplot (ax1)
    ax1.scatter(X, y, alpha=0.6, s=10, label="Data")
    # ax1.plot(X, yPredRef, color="red", label="Ref Line", linewidth=3)
    ax1.plot(X, yPred, color="orange", label=f"{currentOptimizer}", linewidth=2)

    # Plot each line from historyParams on ax1
    for i, params in enumerate(historyParams):
        if i % 10 != 0:
            continue

        m, b = params
        yPredHistory = X * m + b  # Calculate predicted y for each parameter set
        ax1.plot(X, yPredHistory, color="orange", linewidth=0.5)  # Thin line for history

    # Customize the first subplot (regression lines)
    ax1.set_title(f"Linear Regression with {currentOptimizer}")
    ax1.set_xlabel("X")
    ax1.set_ylabel("y")
    ax1.legend()

    # Plot the error (MSE) on the second subplot (ax2)
    ax2.plot(range(len(mse)), mse, color="green", label="MSE")
    ax2.set_title("Mean Squared Error over epochs")
    ax2.set_xlabel("epochs")
    ax2.set_ylabel("MSE")
    ax2.legend()

    # Adjust layout to make room for both subplots
    plt.tight_layout()

    # Show the plot
    plt.show()

if __name__ == '__main__':
    main()