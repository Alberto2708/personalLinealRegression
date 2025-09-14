import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import learning_curve, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

def plotLinearModel(x, y):

    x = np.array(x).reshape(-1, 1)
    y = np.array(y)


    model = make_pipeline(
    StandardScaler(),             
    LinearRegression(
        tol=1e-06,
        n_jobs=None,
    )
)

    train_sizes, train_scores, val_scores = learning_curve(
        model, x, y, cv=5,
        train_sizes=np.linspace(0.1, 1.0, 5), n_jobs=-1
    )

    plt.figure(figsize=(8,5))
    plt.plot(train_sizes, train_scores, "o-", label="Training R²")
    plt.plot(train_sizes, val_scores, "o-", label="Validation R²")
    plt.xlabel("Tamaño del conjunto de entrenamiento")
    plt.ylabel("R²")
    plt.title("Learning curve scikit-learn linear regression")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid()
    plt.show()

    return None

