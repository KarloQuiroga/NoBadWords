import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import re
import nltk

# utilizar el siguiente comando para descargar los paquetes de nltk, solo la primera vez
# nltk.download()
stremmer = nltk.SnowballStemmer('spanish')


def clean_text(text):
    # Convertir el texto a minúsculas
    text = text.lower()
    # Eliminar caracteres no deseados (puntuación y números)
    text = re.sub(r'[^a-záéíóúñü]+', ' ', text)
    # Eliminar las palabras vacías (stop words)
    words = text.split()
    stopwords = nltk.corpus.stopwords.words('spanish')
    words = [word for word in words if word not in set(stopwords)]
    return ' '.join(words)


df = pd.read_csv('lista_negra.csv', encoding='utf-8')

# Aplicar la función de limpieza a cada fila de la columna de texto
df['cleaned_text'] = df['text'].apply(clean_text)

# Dividir el conjunto de datos en características (X) y etiquetas (y)
X = df['cleaned_text']
y = df['label']

# Dividir en conjuntos de entrenamiento y prueba (80% para entrenamiento, 20% para pruebas)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
# Crear un CountVectorizer y transformar el texto en una matriz numérica
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)

# Crear una instancia de DecisionTreeClassifier y ajustar (entrenar) el modelo con los datos de entrenamiento
logreg = DecisionTreeClassifier()
logreg.fit(X_train_vec, y_train)

X_test_vec = vectorizer.transform(X_test)

# Predecir las etiquetas para el conjunto de datos de prueba
y_pred = logreg.predict(X_test_vec)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)

print(f'La precisión del modelo de árbol de decisión es: {accuracy:.2f}')

print('Ejemplo de predicción:')
test = 'eres un pinche pendejo'
df = vectorizer.transform([test]).toarray()
print(f'Texto: {logreg.predict(df)}')
