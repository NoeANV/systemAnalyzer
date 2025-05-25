import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import joblib

# Leer dataset
df = pd.read_csv('dataset.csv')

# Seleccionar columnas relevantes
df = df[['Product', 'Type', 'Vendor', 'TDP (W)', 'Process Size (nm)', 'Freq (MHz)', 'FP32 GFLOPS']]

# Eliminar filas vacías
df = df.dropna()

# Crear etiquetas de clase en base a FP32 GFLOPS: bajo, medio, alto
df['Performance_Class'] = pd.qcut(df['FP32 GFLOPS'], q=3, labels=['Bajo', 'Medio', 'Alto'])

# Eliminar columna continua original (ya no se usa en clasificación)
df = df.drop(columns=['FP32 GFLOPS'])

# Convertir categóricas a variables dummy
df = pd.get_dummies(df, columns=['Product', 'Type', 'Vendor'], drop_first=True)

# Separar características (X) y etiquetas (y)
X = df.drop(columns=['Performance_Class'])
y = df['Performance_Class']

# Dividir en entrenamiento y prueba (con estratificación)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Entrenar modelo de clasificación
modelo = RandomForestClassifier(random_state=42)
modelo.fit(X_train, y_train)

# Guardar modelo entrenado y columnas
joblib.dump(modelo, 'modelo_clasificacion.pkl')
joblib.dump(X.columns.tolist(), 'modelo_columnas_clasificacion.pkl')

# Evaluación del modelo
y_pred = modelo.predict(X_test)

print("\n📊 Reporte de clasificación:\n")
print(classification_report(y_test, y_pred))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=modelo.classes_)
disp.plot()
plt.title("Matriz de Confusión")
plt.show()

print("✅ Modelo de clasificación entrenado y guardado como 'modelo_clasificacion.pkl'")
