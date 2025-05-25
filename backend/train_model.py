import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Leer el dataset
df = pd.read_csv('dataset.csv')

# Elegimos columnas relevantes (incluyendo las nuevas)
df = df[['Product', 'Type', 'Vendor', 'TDP (W)', 'Process Size (nm)', 'Freq (MHz)', 'FP32 GFLOPS']]

# Convertir las columnas categóricas a variables dummy
df = pd.get_dummies(df, columns=['Product', 'Type', 'Vendor'], drop_first=True)

# Eliminar filas con datos faltantes
df = df.dropna()

# Separar datos
X = df.drop(columns=['FP32 GFLOPS'])  # Entrada (todas las columnas excepto la salida)
y = df['FP32 GFLOPS']                # Salida (potencia estimada)

# Separar en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Entrenar modelo de regresión
modelo = RandomForestRegressor()
modelo.fit(X_train, y_train)

# Guardar modelo y columnas utilizadas
joblib.dump(modelo, 'modelo.pkl')
joblib.dump(X.columns.tolist(), 'modelo_columnas.pkl') 


print("Modelo entrenado y guardado como modelo.pkl")
