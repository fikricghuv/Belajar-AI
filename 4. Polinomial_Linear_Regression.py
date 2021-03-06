""" Linear Regression Polinomial (fungsi kuadratik)
dengan fungsi :
    y = a0 + a1 * x1 + a2 * x2 ^2 + a3 * x3 ^3 ....

sama dengan Simple linear regression yang membedakan adalah saat data 
dianggap kurang fit saat di coba dengan Simple linear.
"""
# Mengimpor library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Mengimpor dataset
dataset = pd.read_csv('data/Posisi_gaji.csv')
print(dataset)
X = dataset.iloc[: , 1:2].values #varibel X harus dibuat menjadi matrix
y = dataset.iloc[: , 2].values

# Fitting Linear Regression ke dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
 

# Fitting Polynomial Regression ke dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)  
# nantinya degree diganti menjadi 4 untuk meningkatkan akurasi
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
 
# Visualisasi hasil regresi sederhana
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Sesuai atau tidak (Linear Regression)')
plt.xlabel('Level posisi')
plt.ylabel('Gaji')
plt.show()
 
# Visualisasi hasil regresi polynomial
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(X_poly), color = 'blue')
plt.title('Sesuai atau tidak (Polynomial Regression)')
plt.xlabel('Level posisi')
plt.ylabel('Gaji')
plt.show()
 
 
# Memprediksi hasil dengan regresi sederhana
lin_reg.predict(np.array([[6.5]]))
 
# Memprediksi hasil dengan regresi polynomial
lin_reg_2.predict(poly_reg.fit_transform(np.array([[6.5]])))




