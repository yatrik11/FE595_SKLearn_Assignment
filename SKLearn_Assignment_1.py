# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:11:16 2019

@author: Yatri Kalathia
"""

#SKLearn assignment Question 1

import pandas as pd
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.datasets import load_boston
housing_data = load_boston()

#checking for keys since the data variable itself is a disctionary
print(housing_data.keys())

#checking the size of dataset
print(housing_data.data.shape)

print(housing_data.feature_names)

print(housing_data.DESCR)

#converting into pandas dataframe
boston_data = pd.DataFrame(housing_data.data)
print(boston_data.head())

boston_data.columns = housing_data.feature_names
print(boston_data.head())

boston_data['PRICE'] = housing_data.target
print(boston_data.head())

print(boston_data.describe())

X = pd.DataFrame(boston_data,columns=['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT'])
Y = boston_data['PRICE']

# Linear Regression model and accuracy

lin_model = LinearRegression()
lin_model.fit(X, Y)

#finding the most optimal co-efficients
coeff_df = pd.DataFrame(lin_model.coef_, X.columns, columns=['Coefficient'])
coeff_df

#This means that for a unit increase in nitric oxides concentration, there is a decrease of 17.76 units in "Price".
#Similarly, a unit decrease in “RM“ results in an increase of 3.80 units in "Price".
#We can see that the rest of the features have very little effect on the value of houses in Boston.

#So, we can conclude that NOX has most negative impact on the price of boston housing and RM has most positive impact.