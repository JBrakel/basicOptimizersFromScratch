import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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


class Optimizer():
    def __init__(self, X, y, learningRate=0.01, epochs=50):
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

    def clearLineParamsAndErrors(self):
        self.historyParams, self.mse = [], []

    def computeGradient(self, w, batchSize=None):
        if batchSize is None:
            X, y = self.X, self.y
        else:
            idx = np.random.randint(0, len(self.X), batchSize)
            X, y = np.take(self.X, idx), np.take(self.y, idx)

        N = len(X)
        m, b = w
        yPred = X * m + b
        error = yPred - y

        # Gradient of MSE with respect to w and b
        dm = (2 / N) * np.dot(X.flatten(), error.flatten())
        db = (2 / N) * np.sum(error)
        dw = np.array([dm, db])

        # Store values to plot
        self.saveLineParamsAndErrors(w, np.mean(np.square(error)))
        return dw

    def predict(self, currentOptimizer, batchSize=None, rho=0.9):
        # Run the selected optimizer
        if currentOptimizer == "gd":
            X, yPred, m, b = self.gradientDescent()
        elif currentOptimizer == "sgd":
            X, yPred, m, b = self.stochasticGradientDescent(batchSize)
        elif currentOptimizer == "sgdMom":
            X, yPred, m, b = self.stochasticGradientDescentAndMomentum(batchSize, rho)
        elif currentOptimizer == "sgdNesMom":
            X, yPred, m, b = self.stochasticGradientDescentAndNesterovMomentum(batchSize, rho)
        else:
            raise ValueError("Invalid optimizer type.")

        # Get the history before clearing it
        history, mse = self.getLineParamsAndErrors()

        # Clear previous history for future runs
        self.clearLineParamsAndErrors()

        return X, yPred, m, b, history, mse

    def gradientDescent(self):
        w = self.initWeights()
        for _ in range(self.epochs):
            dw = self.computeGradient(w)
            w = w - self.learningRate * dw

        self.mPred, self.bPred = w
        yPred = w[0] * self.X + w[1]
        return self.X, yPred, w[0], w[1]

    def stochasticGradientDescent(self, batchSize):
        w = self.initWeights()
        for _ in range(self.epochs):
            dw = self.computeGradient(w, batchSize)
            w = w - self.learningRate * dw

        self.mPred, self.bPred = w
        yPred = w[0] * self.X + w[1]
        return self.X, yPred, w[0], w[1]

    def stochasticGradientDescentAndMomentum(self, batchSize, rho=0.9):
        w = self.initWeights()
        v = 0
        for _ in range(self.epochs):
            dw = self.computeGradient(w, batchSize)
            v = rho * v + dw
            w = w - self.learningRate * v

        self.mPred, self.bPred = w
        yPred = w[0] * self.X + w[1]
        return self.X, yPred, w[0], w[1]

    def stochasticGradientDescentAndNesterovMomentum(self, batchSize, rho=0.9):
        w = self.initWeights()
        v = 0
        for _ in range(self.epochs):
            dw = self.computeGradient(w, batchSize)
            old_v = v
            v = rho * v - self.learningRate * dw
            w = w - rho * old_v - (1-rho) * v

        self.mPred, self.bPred = w
        yPred = w[0] * self.X + w[1]
        return self.X, yPred, w[0], w[1]

def main():
    # Create data points
    X, y, mTrue, bTrue = generateTestSamples(5000)
    print(f"True values: m={round(mTrue, 1)}, b={round(bTrue, 1)}")

    # Create reference model
    _, yPredRef, mRef, bRef = predictWithRefModel(X, y)
    print(f"Ref  values: m={round(float(mRef), 1)}, b={round(float(bRef), 1)}")

    # Choose optimizer
    opt = Optimizer(X, y, learningRate=0.001, epochs=500)

    _, yPredGD, mPredGD, bPredGD, historyParamsGD, mseGD = opt.predict("gd")
    print(f"Pred values: m={round(float(mPredGD), 1)}, b={round(float(bPredGD), 1)}")

    _, yPredSGD, mPredSGD, bPredSGD, historyParamsSGD, mseSGD = opt.predict("sgd", batchSize=200)
    print(f"Pred values: m={round(float(mPredSGD), 1)}, b={round(float(bPredSGD), 1)}")

    _, yPredSGDmom, mPredSGDmom, bPredSGDmom, historyParamsSGDmom, mseSGDmom = opt.predict("sgdMom", batchSize=200, rho=0.9)
    print(f"Pred values: m={round(float(mPredSGDmom), 1)}, b={round(float(bPredSGDmom), 1)}")

    _, yPredSGDnesmom, mPredSGDnesmom, bPredSGDnesmom, historyParamsSGDnesmom, mseSGDnesmom = opt.predict("sgdMom", batchSize=200, rho=0.9)
    print(f"Pred values: m={round(float(mPredSGDnesmom), 1)}, b={round(float(bPredSGDnesmom), 1)}")

    # Output
    # Create a figure with two subplots: one for the regression lines and one for the error
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    # Plot the regression lines on the first subplot (ax1)
    ax1.scatter(X, y, alpha=0.6, s=10, label="Data")
    ax1.plot(X, yPredRef, color="red", label="True Line", linewidth=3)
    ax1.plot(X, yPredGD, color="orange", label="GD", linewidth=2)
    ax1.plot(X, yPredSGD, color="purple", label="SGD", linewidth=2)
    ax1.plot(X, yPredSGDmom, color="#00FF00", label="SGD + Momentum", linewidth=2)
    ax1.plot(X, yPredSGDnesmom, color="brown", label="SGD + Nesterov Momentum", linewidth=2)

    # Plot each line from historyParams on ax1
    for i, params in enumerate(historyParamsGD):
        if i % 10 != 0:
            continue
        m, b = params
        yPredHistory = X * m + b  # Calculate predicted y for each parameter set
        ax1.plot(X, yPredHistory, color="orange", linewidth=1)  # Thin line for history

    for i, params in enumerate(historyParamsSGD):
        if i % 10 != 0:
            continue
        m, b = params
        yPredHistory = X * m + b  # Calculate predicted y for each parameter set
        ax1.plot(X, yPredHistory, color="purple", linewidth=1)  # Thin line for history

    for i, params in enumerate(historyParamsSGDmom):
        if i % 10 != 0:
            continue
        m, b = params
        yPredHistory = X * m + b  # Calculate predicted y for each parameter set
        ax1.plot(X, yPredHistory, color="#00FF00", linewidth=1)  # Thin line for history

    for i, params in enumerate(historyParamsSGDnesmom):
        if i % 10 != 0:
            continue
        m, b = params
        yPredHistory = X * m + b  # Calculate predicted y for each parameter set
        ax1.plot(X, yPredHistory, color="brown", linewidth=1)  # Thin line for history

    ax1.plot(X, yPredRef, color="red", linewidth=3)


    # Customize the first subplot (regression lines)
    ax1.set_title(f"Linear Regression with different Optimizers")
    ax1.set_xlabel("X")
    ax1.set_ylabel("y")
    ax1.legend()

    # Plot the error (MSE) on the second subplot (ax2)
    ax2.plot(range(len(mseGD)), mseGD, color="orange", label="GD")
    ax2.plot(range(len(mseSGD)), mseSGD, color="purple", label="SGD")
    ax2.plot(range(len(mseSGDmom)), mseSGDmom, color="#00FF00", label="SGD + Momentum")
    ax2.plot(range(len(mseSGDnesmom)), mseSGDnesmom, color="brown", label="SGD + Nesterov Momentum")
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