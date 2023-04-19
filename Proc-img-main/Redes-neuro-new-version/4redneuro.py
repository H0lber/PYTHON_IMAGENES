import tensorflow as tf
import cv2
import numpy as np

numeros_prueba = range(0,10)
imagenes = []
etiquetas_imagenes = []

# Leemos y preprocesamos las imágenes de los dígitos
for number in numeros_prueba:
    for i in range(0,280*11,28):
        ini_ver = i 
        fin_ver = i+27
        for j in range (0,280,28):
            ini_hor = j
            fin_hor = j+27
            imgReaded = f'grilla{str(number)}.jpg'
            img = cv2.imread(imgReaded, cv2.IMREAD_GRAYSCALE)
            img_numero = img[ini_ver:fin_ver, ini_hor:fin_hor].copy()
            img_numero = cv2.resize(img_numero, (28, 28))
            img_numero = cv2.bitwise_not(img_numero)
            imagenes.append(img_numero)
            etiquetas_imagenes.append(number)

imagenes = np.array(imagenes)
etiquetas_imagenes = np.array(etiquetas_imagenes)

modelo = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)), # Capa de entrada que aplana las imágenes de 28x28 píxeles a un vector de 784 elementos
    tf.keras.layers.Dense(2000, activation='relu'), # Capa oculta con 2000 neuronas y función de activación ReLU
    tf.keras.layers.Dense(1500, activation='relu'), # Capa oculta con 1500 neuronas y función de activación ReLU
    tf.keras.layers.Dense(1116, activation='relu'), # Capa oculta con 1116 neuronas y función de activación ReLU
    tf.keras.layers.Dense(10) # Capa de salida con 10 neuronas (una por cada número)
])

# Compilamos el modelo con una función de pérdida y un optimizador
modelo.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Ajustamos el modelo a las imágenes de entrenamiento
modelo.fit(imagenes, etiquetas_imagenes, epochs=25)

# Evaluamos el modelo en las imágenes de prueba
resultado = modelo.evaluate(imagenes, etiquetas_imagenes)
print('Pérdida en el conjunto de prueba:', resultado[0])
print('Precisión en el conjunto de prueba:', resultado[1])

# Probamos el modelo con algunas imágenes nuevas
print('Real: 1')
img_nueva = cv2.imread("1-2prueba.jpg", 0)
img_nueva = cv2.resize(img_nueva, (28, 28))
img_nueva = cv2.bitwise_not(img_nueva)
img_nueva = img_nueva.astype('float32') / 255
img_nueva = img_nueva.reshape(1, 28, 28)
prediccion = modelo.predict(img_nueva)
numero_predicho = np.argmax(prediccion)
print('El número predicho es:', numero_predicho)

print('Real: 3')
img_nueva = cv2.imread("3-2prueba.jpg", 0)
img_nueva = cv2.bitwise_not(img_nueva)
img_nueva = cv2.resize(img_nueva, (28, 28))

#Escalar pixeles
img_nueva = img_nueva.astype('float32')
img_nueva /= 255

#Redimensionar a un arreglo de 4 dimensiones
img_nueva = np.expand_dims(img_nueva, axis=0)
img_nueva = np.expand_dims(img_nueva, axis=3)

#Obtener la predicción del modelo
prediccion = model.predict(img_nueva)
numero = np.argmax(prediccion)

print(f'Predicción: {numero}')
