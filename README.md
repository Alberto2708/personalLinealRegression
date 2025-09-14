

# Modelo de regresión lineal realizado desde scratch VS. LinearRegression de scikitlearn


## Dependencias necesarias

Para poder correr el código es recomendable tener un entorno virtual de python en el que se instalen las siguientes dependencias:


``` pip install numpy pandas matplotlib scikitlearn ```

## Estructura del código 


```plaintext
    personalLinealRegression/
    ├── libraryModel/
    │   └── linearModel.py
    |
    ├── personalModel/
    │   |── betModel.py
    |   |── betoModelOOP.py
    |   └── learningCurvePersonal.py
    |
    ├── .gitignore
    |── README.md
    └── betoModelTest.py

```

## ¿Que contiene cada archivo?

- linearModel: La definición de un modelo de regresión lineal usando scikit-learn
- betModel: Modelo linear personal definido en forma de funciones 
- betoModelOOP: En base al modelo de regresión linear ya definido, este archivo es su definición en POO
- learningCurvePersonal: Una función para evaluar el desempeño de mi modelo personal 
- betoModelTest: Se agarra un dataset de pingüinos y se mandan a llamar los demás modelos para crear dos gráficas de evaluación basadas en R2

## ¿Como correr el programa?

Asegurandote que estás en el directorio root
En windows:
``` python .\betoModelTest.py ```