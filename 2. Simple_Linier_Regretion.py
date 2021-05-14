""" PERSAMAAN MATEMATIS 
y = B0 + B1 * Xi
y = Variable Dependen
B0 = konstanta
B1 = koefisien untuk Xi
Xi = Variabel independen
"""
# Import library yang dibutuhkan
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#import data set
dataset = pd.read_csv('data/daftar_gaji.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values
print(dataset)

#membagi menjadi Training dan Test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=1/3 , random_state= 0)


# Fitting simple linier regration terhadap Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Memprediksi hasil test set
Y_pred = regressor.predict(X_test)

# Visualisasi hasil training set
plt.scatter(X_train, Y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Gaji vs Pengalaman (Training Set)')
plt.xlabel('Tahun Bekerja')
plt.ylabel('Gaji')
plt.show()

# Visualisasi hasil test set
plt.scatter(X_test, Y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Gaji vs Pengalaman (Training Set)')
plt.xlabel('Tahun Bekerja')
plt.ylabel('Gaji')
plt.show()








