import pandas as pd 
import numpy as np


#En este arcihvo están todas las funciones necesarias

# Función de covarianza y correlación

def covariance(x, y):
    y_mean = y.mean()
    x_mean = x.mean()
    cov = 0
    for i in range(len(x)):
        cov += (x[i] - x_mean) * (y[i] - y_mean)

    return cov / (len(x) - 1)

def correlation(x, y):
    return covariance(x, y) / (x.std() * y.std())

def standardize(x):
    return (x - x.mean()) / x.std()


# Función predict y función costo 

def predict(x, w, b):
    return w * x + b


def mse(y, y_pred):
    return ((y - y_pred) ** 2).mean()


# Función descenso de gradiente w y b 

def b_gradient_descent(x, w, b, lr,y):
    n = len(x)
    partial_b_sum = 0
    for i in range(0, n):
        partial_b_sum += (y.iloc[i] - b - (w*x.iloc[i])) * (-1)

    partial_b = partial_b_sum * (2/n)

    return b - lr*(partial_b)



def w_gradient_descent(x, w, b, lr, y):
    n = len(x)
    partial_w_sum = 0
    for i in range(0, n):
        partial_w_sum += (y.iloc[i]-b-(w*x.iloc[i])) * ((-1)*x.iloc[i])
    partial_w = partial_w_sum * (2/n)

    return w - lr * partial_w


#Función general

def trainModel(x, y, lr, max_iter):
    tol = 1e-6
    if len(x) != len(y):
        print("size of arrays for features and targets do not match")
        return

    np.random.seed(42)
    w = 0.1
    b = 0.1
    model_cost = float("inf")
    iter_cost = float("inf")

    for i in range(0, max_iter):
        predictions=[]

        for j in range(0, len(x)):
            predictions.append(predict(x.iloc[j], w, b))

        print(f"current cost: {iter_cost}")
        iter_cost = mse(y, predictions)
        if abs(model_cost - iter_cost) < tol: 
            break
        model_cost = iter_cost
        w = w_gradient_descent(x, w, b, lr, y)
        b = b_gradient_descent(x, w, b, lr, y)

    return w, b

def testModel(w, b, x, y):
    predictions = []
    for i in range(0, len(x)):
        predictions.append(predict(x.iloc[i], w, b))
    
    results = mse(y, predictions)

    return results


#Funcion para entrenar y testear 

def linearModel(x_train, x_test, y_train, y_test, lr=0.1, max_iter=100):
    print("Beggining training")
    print(f"Max iterations: {max_iter}")
    print(f"Learning rate: {lr}")
    x_train = standardize(x_train)
    x_test = standardize(x_test)
    w, b = trainModel(x_train, y_train, lr, max_iter)

    results = testModel(w, b, x_test, y_test)

    print("Final results:")
    print(f"y = {np.mean(w)}x + {np.mean(b)}")
    print(f"Final MSE: {np.mean(results)}")
