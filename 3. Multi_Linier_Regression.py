"""
     Persamaan Matematis
y = B0 + B1 * X1 + B2 * X2 + B3 * X3 + .. + Bn * Xn
Multi Liear Regression memiliki variabel independen sebanyak n.

1. Variabel dependen = Profit (data numerik)
2. Variabel Independen = - Biaya R&D (data numerik)
                         - Biaya Administrasi (data numerik)
                         - Biaya Marketing (data numerik)
                         - Wilayah perusahaan (data kategori)
                         
Fungsinya matematis nya seperti ini :
    profit = a0 + a1 * R&D + a2 * marketing + a3 * administrasi + a4 * wilayah + a5 *wilayah2
    
    
Syarat agar model yang digunakan sesuai dengan Metode Multi Linear Regression :
    1. Linearity
    2. Homoscedasticity
    3. Multivariate normality
    4. Independence of errors
    5. Lack of multicollinearity
"""

#Import Liberary
import numpy as np
import pandas as pd

#mengimport data set
dataset = pd.read_csv('data/50_startups.csv')
print(dataset)
X = dataset.iloc[: , :-1].values
Y = dataset.iloc[:, 4].values


#Encodde data kategori
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
transformer = ColumnTransformer(
        [('encoder', OneHotEncoder(), [3])],
        remainder='passthrough')
X = np.array(transformer.fit_transform(X), dtype=np.float)
 
# Menghindari jebakan dummy variabel
X = X[:, 1:]

# Membagi data menjadi training dan test
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train , Y_test = train_test_split(X, Y, test_size = 0.2 , random_state = 0)

# membuat model multiple Linear Regression dari training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

#Memprediksi hasil tes set
y_pred = regressor.predict(X_test)

# Memilih model multiple regresi yang paling baik dengan metod
import statsmodels.api as sm
X_new = sm.add_constant(X)
X_opt = X_new[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X_new[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X_new[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X_new[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()
X_opt = X_new[:, [0, 3]]
regressor_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
regressor_OLS.summary()


# Secara otomatis mengeliminasi  model yang tepat

import statsmodels.api as sm
def backwardElimination(x, SL):
    jum_kol = len(x[0])
    for i in range(0, jum_kol):
        regressor_OLS = sm.OLS(endog = Y, exog = X).fit()
        p_val = regressor_OLS.pvalues.astype(float)
        max_index = np.argmax(p_val, axis = 0)
        nilai_max = max(regressor_OLS.pvalues).astype(float)
        if nilai_max > SL:
            x = np.delete(x, max_index, 1)
    print(regressor_OLS.summary())

SL =0.05
X_new = sm.add_constant(X)
X_opt = X_new[:, [0,1,2,3,4,5]]
X_Modeled = backwardElimination(X_opt, SL)



















