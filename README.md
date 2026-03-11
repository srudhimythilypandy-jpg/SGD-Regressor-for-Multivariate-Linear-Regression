# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Input the dataset containing house features such as area, bedrooms, and bathrooms.
   
2.Preprocess the data by splitting it into training and testing sets and applying feature scaling.

3.Train the SGD Regressor model using the training data to learn the relationship between inputs and outputs.

4.Predict the house price and number of occupants for new data and evaluate the model performance.

## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: srudhi.m
RegisterNumber: 25016156

import numpy as np
import pandas as pd
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler

np.random.seed(0)
X = np.random.rand(100, 3)  # 100 samples, 3 features
true_weights = [3, 5, 2]
Y = X @ true_weights + 4 + np.random.randn(100)  # Y = 3x1 + 5x2 + 2x3 + 4 + noise

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SGDRegressor(max_iter=1000, learning_rate='invscaling', eta0=0.01, random_state=42)
model.fit(X_train_scaled, Y_train)

Y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred

print("Coefficients (weights):", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mse)
print("R² Score:", r2)



*/
```

## Output:
<img width="617" height="91" alt="image" src="https://github.com/user-attachments/assets/5f922900-c86b-4197-aa75-17c1c4e70a9e" />



## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
