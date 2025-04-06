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
        return np.array([mInit, bInit])

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

        yPred = X * w[0] + w[1]
        error = yPred - y

        # Gradient of MSE with respect to w and b
        dm = (2 / len(X)) * np.dot(X.flatten(), error.flatten())
        db = (2 / len(X)) * np.sum(error)

        # Store values to plot
        self.saveLineParamsAndErrors(w, np.mean(np.square(error)))
        return np.array([dm, db])

    def gradientDescent(self):
        w = self.initWeights()
        for _ in range(self.epochs):
            dw = self.computeGradient(w)
            w = w - self.learningRate * dw

        yPred = w[0] * self.X + w[1]
        return self.X, yPred, w[0], w[1]

    def stochasticGradientDescent(self, batchSize):
        w = self.initWeights()
        for _ in range(self.epochs):
            dw = self.computeGradient(w, batchSize)
            w = w - self.learningRate * dw

        yPred = w[0] * self.X + w[1]
        return self.X, yPred, w[0], w[1]

    def momentum(self, batchSize, rho):
        w = self.initWeights()
        v = 0
        for _ in range(self.epochs):
            dw = self.computeGradient(w, batchSize)
            v = rho * v + dw
            w = w - self.learningRate * v

        yPred = w[0] * self.X + w[1]
        return self.X, yPred, w[0], w[1]

    def nesterovMomentum(self, batchSize, rho):
        w = self.initWeights()
        v = 0
        for _ in range(self.epochs):
            dw = self.computeGradient(w, batchSize)
            old_v = v
            v = rho * v - self.learningRate * dw
            w -= rho * old_v - (1+rho) * v

        yPred = w[0] * self.X + w[1]
        return self.X, yPred, w[0], w[1]

    def adagrad(self, batchSize):
        w = self.initWeights()
        gradSquared = np.zeros_like(w)
        for _ in range(self.epochs):
            dw = self.computeGradient(w, batchSize)
            gradSquared = gradSquared + dw * dw
            w = w - self.learningRate * dw / (np.sqrt(gradSquared) + 1e-7)

        yPred = w[0] * self.X + w[1]
        return self.X, yPred, w[0], w[1]

    def RMSprop(self, batchSize, rho):
        w = self.initWeights()
        gradSquared = np.zeros_like(w)
        for _ in range(self.epochs):
            dw = self.computeGradient(w, batchSize)
            gradSquared = rho * gradSquared + (1 - rho) * (dw ** 2)
            w = w - self.learningRate * dw / (np.sqrt(gradSquared) + 1e-7)

        yPred = w[0] * self.X + w[1]
        return self.X, yPred, w[0], w[1]

    def adam(self, batchSize, beta1, beta2, epsilon):
        w = self.initWeights()
        moment1, moment2 = np.zeros_like(w), np.zeros_like(w)
        for t in range(1, self.epochs + 1):
            dw = self.computeGradient(w, batchSize)
            moment1 = beta1 * moment1 + (1 - beta1) * dw
            moment2 = beta2 * moment2 + (1 - beta2) * (dw ** 2)

            # Bias correction
            m_hat = moment1 / (1 - beta1 ** t)
            v_hat = moment2 / (1 - beta2 ** t)

            w = w - self.learningRate * m_hat / (np.sqrt(v_hat) + epsilon)

        yPred = w[0] * self.X + w[1]
        return self.X, yPred, w[0], w[1]

    def predict(self, batchSize=None, rho=0.9, beta1=0.9, beta2=0.9, epsilon=1e-7):
        """
        Run the selected optimization algorithm and return the model predictions, parameters, and history.

        Args:
            batchSize (int, optional): Size of the mini-batch for SGD-based optimizers.
            rho (float, optional): Momentum term for momentum-based methods (default is 0.9).
            beta1 (float, optional): First moment estimate decay for Adam (default is 0.9).
            beta2 (float, optional): Second moment estimate decay for Adam (default is 0.9).
            epsilon (float, optional): Small value to avoid division by zero in Adam (default is 1e-7).

        Returns:
            X (array): Input data.
            yPred (array): Predicted values.
            m (float): Slope of the regression line.
            b (float): Intercept of the regression line.
            history (list): History of model parameters.
            mse (list): Mean squared error history.
        """
        if self.currentOptimizer == "gd":
            X, yPred, m, b = self.gradientDescent()
        elif self.currentOptimizer == "sgd":
            X, yPred, m, b = self.stochasticGradientDescent(batchSize)
        elif self.currentOptimizer == "momentum":
            X, yPred, m, b = self.momentum(batchSize, rho)
        elif self.currentOptimizer == "nesterov":
            X, yPred, m, b = self.nesterovMomentum(batchSize, rho)
        elif self.currentOptimizer == "adagrad":
            X, yPred, m, b = self.adagrad(batchSize)
        elif self.currentOptimizer == "rmsprop":
            X, yPred, m, b = self.RMSprop(batchSize, rho)
        elif self.currentOptimizer == "adam":
            X, yPred, m, b = self.adam(batchSize, beta1, beta2, epsilon)
        else:
            raise ValueError("Invalid optimizer type.")

        # Get the history before clearing it
        history, mse = self.getLineParamsAndErrors()

        # Clear previous history for future runs
        self.clearLineParamsAndErrors()

        return X, yPred, m, b, history, mse


def main():
    # Create test data with 5000 samples
    X, y, mTrue, bTrue = generateTestSamples(5000)

    # Generate predictions using the reference model
    _, yPredRef, mRef, bRef = predictWithRefModel(X, y)

    # Define a dictionary for optimizer names to make it more readable
    dict = {
        "gd": "Gradient Descent",
        "sgd": "Stochastic Gradient Descent",
        "momentum": "Momentum Update",
        "nesterov": "Nesterov Momentum Update",
        "adagrad": "Adagrad",
        "rmsprop": "RMSProp",
        "adam": "Adam"
    }

    # Set the current optimizer and parameters for the experiments
    currentOptimizer = "adam"  # Can change this to experiment with different optimizers
    epochs = 300  # Number of iterations for training
    lr1, lr2, lr3 = 0.1, 0.1, 0.1  # Learning rates
    bs1, bs2, bs3 = 150, 150, 150  # Batch sizes
    rho1, rho2, rho3 = 0.5, 0.9, 0.99  # Values for rho (momentum parameter) in optimizers
    beta1_1, beta1_2, beta1_3 = 0.99, 0.9, 0.9  # Beta1 values for Adam optimizer
    beta2_1, beta2_2, beta2_3 = 0.999, 0.95, 0.999  # Beta2 values for Adam optimizer
    epsilon = 1e-7  # Small constant to prevent division by zero in some optimizers

    # Create three instances of the Optimizer class with different parameters and predict
    opt1 = Optimizer(currentOptimizer, X, y, learningRate=lr1, epochs=epochs)
    params1 = opt1.predict(batchSize=bs1, rho=rho1, beta1=beta1_1, beta2=beta2_1, epsilon=epsilon)
    _, yPred1, mPred1, bPred1, historyParams1, mse1 = params1

    opt2 = Optimizer(currentOptimizer, X, y, learningRate=lr2, epochs=epochs)
    params2 = opt2.predict(batchSize=bs2, rho=rho2, beta1=beta1_2, beta2=beta2_2, epsilon=epsilon)
    _, yPred2, mPred2, bPred2, historyParams2, mse2 = params2

    opt3 = Optimizer(currentOptimizer, X, y, learningRate=lr3, epochs=epochs)
    params3 = opt3.predict(batchSize=bs3, rho=rho3, beta1=beta1_3, beta2=beta2_3, epsilon=epsilon)
    _, yPred3, mPred3, bPred3, historyParams3, mse3 = params3

    # Create a figure with two subplots: one for the regression lines and one for the error (MSE)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    # Plot the regression lines for each optimizer and its respective history
    # For historyParams1 (results from the first optimizer)
    for i, params in enumerate(historyParams1[:-1]):
        if i % 10 != 0:
            continue
        m, b = params
        yPredHistory = X * m + b
        ax1.plot(X, yPredHistory, color="#00FF00", linewidth=0.3)

    # For historyParams2 (results from the second optimizer)
    for i, params in enumerate(historyParams2[:-1]):
        if i % 10 != 0:
            continue
        m, b = params
        yPredHistory = X * m + b
        ax1.plot(X, yPredHistory, color="orange", linewidth=0.3)

    # For historyParams3 (results from the third optimizer)
    for i, params in enumerate(historyParams3[:-1]):
        if i % 10 != 0:
            continue
        m, b = params
        yPredHistory = X * m + b
        ax1.plot(X, yPredHistory, color="red", linewidth=0.15)

    # Plot the final regression line on the first subplot (ax1) for each optimizer
    label1 = f"Learning Rate: {lr1}, Batch Size: {bs1}, rho: {rho1}, Beta 1: {beta1_1}, Beta 2: {beta2_1}"
    label2 = f"Learning Rate: {lr2}, Batch Size: {bs2}, rho: {rho2}, Beta 1: {beta1_2}, Beta 2: {beta2_2}"
    label3 = f"Learning Rate: {lr3}, Batch Size: {bs3}, rho: {rho3}, Beta 1: {beta1_3}, Beta 2: {beta2_3}"

    # Scatter plot of the data points
    ax1.scatter(X, y, alpha=0.6, s=10, label="Data Points")
    ax1.plot(X, yPred1, color="#00FF00", label=label1, linewidth=2)
    ax1.plot(X, yPred2, color="orange", label=label2, linewidth=2)
    ax1.plot(X, yPred3, color="red", label=label3, linewidth=2)

    # Customize the first subplot (regression lines)
    ax1.set_title(f"Linear Regression with {dict[currentOptimizer]}")
    ax1.set_xlabel("X")
    ax1.set_ylabel("y")
    ax1.legend()

    # Plot the error (MSE) on the second subplot (ax2)
    ax2.plot(range(len(mse1)), mse1, color="#00FF00", label=label1)
    ax2.plot(range(len(mse2)), mse2, color="orange", label=label2)
    ax2.plot(range(len(mse3)), mse3, color="red", label=label3)

    # Customize the second subplot (MSE plot)
    ax2.set_title("Mean Squared Error")
    ax2.set_xlabel("epochs")
    ax2.set_ylabel("MSE")
    ax2.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()