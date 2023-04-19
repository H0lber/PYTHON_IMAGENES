import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.cvtColor(cv2.imread('sponge.jpg'), cv2.COLOR_BGR2RGB) #cargar imagen original y conversion de espacio



reduccion = 50 #porcentaje reduccion
redu_size = float(reduccion/100) #conversion de unidades
img2= cv2.resize(img, (0, 0), fx=0.5, fy=0.5) #reducir imagen y asignar a nueva
superior = inferior = (img.shape[0] - img2.shape[0]) // 2 #calcular bordes del eje vertical
izquierdo = derecho = (img.shape[1] - img2.shape[1]) // 2 #calcular bordes del eje horizontal
img2bord = cv2.copyMakeBorder(img2, superior, inferior, izquierdo, derecho, cv2.BORDER_CONSTANT, value=[255, 255, 255]) #aplicar bordes para visualizar

img3 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) #imagen blanco y negro

img4 = cv2.flip(img, 1) #imagen invertida por eje vertical


edge_kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]) #crear edge_kernel
img5 = cv2.filter2D(img2, -1, edge_kernel)#aplicacion de bordes
img5bord = cv2.copyMakeBorder(img5, superior, inferior, izquierdo, derecho, cv2.BORDER_CONSTANT, value=[255, 255, 255])

# Creacion del plot 
fig, axs = plt.subplots(nrows = 4, ncols = 2, figsize = (8, 8))

axs[0][0].imshow(img)
axs[0][0].axis("off")
axs[0][0].set_title('Imagen Original')

axs[0][1].imshow(img2bord)
axs[0][1].axis("off")
axs[0][1].set_title('Imagen reducida 50%')

axs[1][0].imshow(img)
axs[1][0].axis("off")

axs[1][1].imshow(img3, cmap="gray")
axs[1][1].axis("off")
axs[1][1].set_title('Imagen Escala Grises')

axs[2][0].imshow(img)
axs[2][0].axis("off")

axs[2][1].imshow(img4)
axs[2][1].axis("off")
axs[2][1].set_title('Imagen invertida eje vertical')

axs[3][0].imshow(img)
axs[3][0].axis("off")

axs[3][1].imshow(img5bord)
axs[3][1].axis("off")
axs[3][1].set_title('Imagen reducida con deteccion de bordes')

plt.show()