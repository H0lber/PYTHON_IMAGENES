import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('yeah.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

img2 = cv2.imread('yeah.jpg',0)



# Reducir tamaño a la mitad
img3 = cv2.resize(img_rgb, (0, 0), fx=0.5, fy=0.5)

# Calcular el tamaño del borde
top = bottom = (img.shape[0] - img3.shape[0]) // 2
left = right = (img.shape[1] - img3.shape[1]) // 2

# Agregar borde blanco a la imagen reducida
bordered_img3 = cv2.copyMakeBorder(img3, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])



# Reducir tamaño de imagen a blanco y negro a la mitad
img4 = cv2.resize(img2, (0, 0), fx=0.5, fy=0.5)

# Calcular el tamaño del borde
top = bottom = (img2.shape[0] - img4.shape[0]) // 2
left = right = (img2.shape[1] - img4.shape[1]) // 2

# Agregar borde blanco a la imagen reducida
bordered_img4 = cv2.copyMakeBorder(img4, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])


# Invertir imagen por el eje vertical
img5= cv2.flip(img_rgb, 0)


# Deteccion de bordes
edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
img6 = cv2.filter2D(img2, -1, edge_kernel)

# Reducir tamaño a la mitad de imagen con filtro de deteccion de bordes
img7 = cv2.resize(img6, (0, 0), fx=0.5, fy=0.5)

# Calcular el tamaño del borde
top = bottom = (img2.shape[0] - img7.shape[0]) // 2
left = right = (img2.shape[1] - img7.shape[1]) // 2

# Agregar borde blanco a la imagen reducida
bordered_img7 = cv2.copyMakeBorder(img7, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[255, 255, 255])



fig, axs = plt.subplots(nrows=4, ncols=2, figsize=(12, 6))

axs[0][0].imshow(img_rgb, cmap="gray")
axs[0][0].axis("off")
axs[0][0].set_title('Imagen original')

axs[0][1].imshow(bordered_img3, cmap="gray")
axs[0][1].axis("off")
axs[0][1].set_title('Imagen reducida a la mitad')


axs[1][0].imshow(img_rgb, cmap="gray")
axs[1][0].axis("off")
axs[1][0].set_title('Imagen original')

axs[1][1].imshow(bordered_img4, cmap="gray")
axs[1][1].axis("off")
axs[1][1].set_title('Imagen (BYN) reducida a la mitad')


axs[2][0].imshow(img_rgb, cmap="gray")
axs[2][0].axis("off")
axs[2][0].set_title('Imagen original')

axs[2][1].imshow(img5, cmap="gray")
axs[2][1].axis("off")
axs[2][1].set_title('Imagen invertida verticalmente')


axs[3][0].imshow(img_rgb, cmap="gray")
axs[3][0].axis("off")
axs[3][0].set_title('Imagen original')

axs[3][1].imshow(bordered_img7, cmap="gray")
axs[3][1].axis("off")
axs[3][1].set_title('Imagen con deteccion de bordes reducida')

print(img.shape)
print(img3.shape)


plt.show()













