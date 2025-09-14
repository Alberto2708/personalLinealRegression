import numpy as np


class LinearModel:
    def __init__(self, X_train, y_train, X_test, y_test, lr=0.01, n_iters=1000):
        self.w = 0.1
        self.b = 0.1
        self.lr = lr
        self.n_iters = n_iters
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test


    def standardize(self, x):
        return (x - x.mean()) / x.std()

    def predict(self, x, w, b ):
        return w * x + b

    def mse(self, y, y_pred):
        return ((y - y_pred) ** 2).mean()
    
    def rsquared(self, y, y_pred):
        ss_total = ((y - y.mean()) ** 2).sum()
        ss_residual = ((y - y_pred) ** 2).sum()
        r2 = 1 - (ss_residual / ss_total)
        return r2

    def b_gradient_descent(self, x, w, b, y):
        n = len(x)
        partial_b_sum = 0
        for i in range(0, n):
            partial_b_sum += (y.iloc[i] - b - (w*x.iloc[i])) * (-1)

        partial_b = partial_b_sum * (2/n)

        return b - self.lr*(partial_b)

    def w_gradient_descent(self, x, w, b, y):
        n = len(x)
        partial_w_sum = 0
        for i in range(0, n):
            partial_w_sum += (y.iloc[i]-b-(w*x.iloc[i])) * ((-1)*x.iloc[i])
        partial_w = partial_w_sum * (2/n)

        return w - self.lr * partial_w

    def testModel(self, X, y, w, b):
        predictions = []
        for i in range(0, len(X)):
            predictions.append(self.predict(X.iloc[i], w, b))

        results = self.rsquared(y, predictions)

        return results
    
    def trainModel(self, x, y):
        tol = 1e-6
        if len(x) != len(y):
            print("size of arrays for features and targets do not match")
            return

        np.random.seed(42)
        w = self.w
        b = self.b
        model_cost = float("inf")
        iter_cost = float("inf")

        for i in range(0, self.n_iters):
            predictions=[]

            for j in range(0, len(x)):
                predictions.append(self.predict(x.iloc[j], w, b))

            iter_cost = self.mse(y, predictions)
            if abs(model_cost - iter_cost) < tol: 
                break
            model_cost = iter_cost
            w = self.w_gradient_descent(x, w, b, y)
            b = self.b_gradient_descent(x, w, b, y)

        return w, b
    
    def fit(self):
        print("Beginning training")
        print(f"Max iterations: {self.n_iters}")
        print(f"Learning rate: {self.lr}")
        x_train = self.standardize(self.X_train)
        x_test = self.standardize(self.X_test)

        w, b = self.trainModel(x_train, self.y_train)

        results = self.testModel(x_test, self.y_test, w, b)

        print("Final results:")
        print(f"y = {np.mean(w)}x + {np.mean(b)}")
        print(f"Final MSE: {np.mean(results)}")
