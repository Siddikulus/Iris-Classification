import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

irisdf = pd.read_csv('iris.data')
irisdf.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'target']

lbenc = LabelEncoder()
irisdf.iloc[:, -1] = lbenc.fit_transform(irisdf.iloc[:, -1])
# print(irisdf.head())

X = irisdf.iloc[:, :-1]
Y = irisdf.iloc[:, -1]

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

# standardize = StandardScaler()
# x_train = standardize.fit_transform(x_train)
# x_test = standardize.transform(x_test)


rfc = RandomForestClassifier(n_estimators=500, random_seed=42)
rfc.fit(x_train, y_train)
y_pred = rfc.predict(x_test)

print(accuracy_score(y_test, y_pred))
