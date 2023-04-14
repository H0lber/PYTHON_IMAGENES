from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.datasets import mnist
from keras.utils import to_categorical
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# Reescala las imágenes para que tengan valores entre 0 y 1
X_train = X_train / 255
X_test = X_test / 255

# Convierte las etiquetas en una matriz codificada en caliente (one-hot)
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Agrega una dimensión extra para representar los canales de color (en este caso, solo hay un canal en escala de grises)
X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1]) 
import numpy as np
from PIL import Image

# Carga una imagen de un número (28x28 píxeles)
img = Image.open('seis.jpg').convert('L')
img = img.resize((28, 28))
img = np.array(img)
img = img.reshape((1, 28, 28, 1))
img = img / 255

# Usa el modelo para predecir el número
prediction = model.predict(img)
print(np.argmax(prediction))