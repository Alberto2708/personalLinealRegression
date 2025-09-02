from betModel import covariance, correlation, standardize, predict 
from betModel import cost, b_gradient_descent, w_gradient_descent, trainModel, testModel, linearModel
import pandas as pd
from sklearn.model_selection import train_test_split

#En este archivo se implementa la prueba del modelo, se pueden elegir diferentes 

#Bases de datos a utilizar
url = "https://raw.githubusercontent.com/allisonhorst/palmerpenguins/c19a904462482430170bfe2c718775ddb7dbb885/inst/extdata/penguins_raw.csv"

penguins = pd.read_csv(url).dropna()
penguins_target = penguins["Body Mass (g)"]
penguins_feature = penguins["Culmen Length (mm)"]
#penguins_feature = standardize(penguins_feature)

X_train, X_test, y_train, y_test = train_test_split(
    penguins_feature, penguins_target, test_size=0.33, random_state=42
)

linearModel(
    X_train,
    X_test,
    y_train,
    y_test,
)

