import numpy as np
from PIL import Image
from tensorflow import keras

# Cargamos la imagen completa
img = Image.open("train-7.jpg")

# Convertimos la imagen a un array numpy
img_array = np.array(img)

# Dividimos la imagen en 100 imágenes de 28x28 pixeles
imgs = []
for i in range(10):
    for j in range(10):
        img_crop = img_array[i*28:(i+1)*28, j*28:(j+1)*28]
        imgs.append(img_crop)

# Convertimos las imágenes en un array numpy
imgs_array = np.array(imgs)

# Cargamos los datos de entrenamiento y pruebas de MNIST
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalizamos los datos
x_train = x_train / 255.0
x_test = x_test / 255.0
imgs_array = imgs_array / 255.0

# Creamos el modelo de red neuronal
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

# Compilamos el modelo
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Entrenamos el modelo con los datos de MNIST
model.fit(x_train, y_train, epochs=5)

# Evaluamos el modelo con los datos de pruebas de MNIST
model.evaluate(x_test, y_test)

# Hacemos las predicciones con las imágenes de 28x28 pixeles
predictions = model.predict(imgs_array)

# Imprimimos las predicciones
for i in range(100):
    prediction = np.argmax(predictions[i])
    print(f"La imagen en la posición ({i//10}, {i%10}) contiene un {prediction}")
