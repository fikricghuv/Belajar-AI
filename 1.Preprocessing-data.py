# Mengimpor library yang diperlukan
import numpy as np
import pandas as pd
 
# Import data ke python
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
 
# Memproses data yang hilang (missing)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan, strategy = 'mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
 
# Encoding data kategori dan variabel independen
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()                     # Bisa dihilangkan, baca pembahasan di bawahnya
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])     # Bisa dihilangkan, baca pembahasan di bawahnya
transformer = ColumnTransformer(
        [('Negara', OneHotEncoder(), [0])],
        remainder='passthrough')
X = np.array(transformer.fit_transform(X), dtype=np.float)
 
# Encode variabel dependen
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)