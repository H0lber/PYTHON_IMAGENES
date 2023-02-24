import cv2
from matplotlib import pyplot as plt

# Leer la imagen y crear una copia
img = cv2.imread('yeah.jpg', 0)
img2 = img.copy()

# Reducción de la gama dinámica de la imagen
(row, col) = img.shape
for i in range(row):
    for j in range(col):
        if (img[i][j] < 10):
            img2[i][j] = 10
        if (img[i][j] > 240):
            img2[i][j] = 240

# Normalización de la intensidad de los píxeles
f_max = img2.max()
f_min = img2.min()
img3 = img2.copy()
for i in range(row):
    for j in range(col):
        img3[i][j] = ((img2[i][j] - f_min) / (f_max - f_min)) * 256

# Crear una figura de subtramas
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(8, 8))

# Mostrar la imagen original
axs[0][0].imshow(img, cmap="gray")
axs[0][0].axis("off")

# Mostrar el histograma de la imagen original
axs[0][1].hist(img.ravel(), 256, [0, 256], color='gray')
axs[0][1].set_title('Histograma de la imagen original')

# Mostrar la imagen normalizada
axs[1][0].imshow(img3, cmap="gray")
axs[1][0].axis("off")

# Mostrar el histograma de la imagen normalizada
axs[1][1].hist(img3.ravel(), 256, [0, 256], color='gray')
axs[1][1].set_title('Histograma de la imagen normalizada')

# Mostrar la figura
plt.show()
