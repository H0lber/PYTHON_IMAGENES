import cv2
import matplotlib.pyplot as plt
import numpy as np

# Cargar la imagen utilizando OpenCV
img = cv2.imread('HB.jpg')

# Convertir la imagen en una matriz NumPy
img_array = np.array(img)

# Cambiar el valor de los píxeles
img_array[0:100, 0:100, :] = [0, 0, 0]

# Mostrar la imagen resultante utilizando Matplotlib
plt.imshow(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
plt.show()
