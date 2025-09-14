import numpy as np
import matplotlib.pyplot as plt
import personalModel.betoModelOOP as model


def learning_curve_linear_model(X, y, test_size=0.2, train_sizes=None, lr=0.01, n_iters=1000):


    if train_sizes is None:
        train_sizes = np.linspace(0.1, 1.0, 5)


    split_idx = int((1 - test_size) * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    train_scores, val_scores = [], []

    for frac in train_sizes:
        n_train = int(frac * len(X_train))


        X_sub, y_sub = X_train[:n_train], y_train[:n_train]

        linear_model = model.LinearModel(X_sub, y_sub, X_test, y_test, lr=lr, n_iters=n_iters)
        w, b = linear_model.trainModel(linear_model.standardize(X_sub), y_sub)


        train_r2 = linear_model.testModel(linear_model.standardize(X_sub), y_sub, w, b)
        train_scores.append(train_r2)

        val_r2 = linear_model.testModel(linear_model.standardize(X_test), y_test, w, b)
        val_scores.append(val_r2)

    # Plot
    plt.figure(figsize=(8,5))
    plt.plot(train_sizes, train_scores, "o-", label="Training R²")
    plt.plot(train_sizes, val_scores, "o-", label="Validation R²")
    plt.xlabel("Fraction of training data")
    plt.ylabel("R²")
    plt.title("Learning Curve for LinearModel")
    plt.legend()
    plt.grid()
    plt.show()

    return train_scores, val_scores
