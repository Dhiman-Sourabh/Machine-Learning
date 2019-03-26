import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#importing the DataSet
ds = pd.read_excel('Real estate valuation data set.xlsx')
X = ds.iloc[:,1:-1].values
y = ds.iloc[:,-1].values

#Spilliting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/5, random_state = 0)

#Fitting the Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
reg = RandomForestRegressor(n_estimators = 350, random_state = 0)
reg.fit(X_train, y_train)

#prediction
y_pred = reg.predict(X_test)

