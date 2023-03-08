import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('yeah.jpg', 0)
img2 = img.copy()

#Sobel Norte
edge_kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
img3 = cv2.filter2D(img2, -1, edge_kernel)

#Sobel Noreste
edge_kernel = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])
img4 = cv2.filter2D(img2, -1, edge_kernel)

#Sobel Este
edge_kernel = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
img5 = cv2.filter2D(img2, -1, edge_kernel)

#Sobel Sureste
edge_kernel = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]])
img6 = cv2.filter2D(img2, -1, edge_kernel)

#Sobel Sur
edge_kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
img7 = cv2.filter2D(img2, -1, edge_kernel)

#Sobel Suroeste
edge_kernel = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]])
img8 = cv2.filter2D(img2, -1, edge_kernel)

#Sobel Oeste
edge_kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
img9 = cv2.filter2D(img2, -1, edge_kernel)

#Sobel Noroeste
edge_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
img10 = cv2.filter2D(img2, -1, edge_kernel)

fig, axs = plt.subplots(nrows=3, ncols=3, figsize=(8, 10))


axs[0][0].imshow(img, cmap="gray")
axs[0][0].axis("off")
axs[0][0].set_title('Imagen original')

axs[0][1].imshow(img3, cmap="gray")
axs[0][1].axis("off")
axs[0][1].set_title('Sobel Norte')

axs[0][2].imshow(img4, cmap="gray")
axs[0][2].axis("off")
axs[0][2].set_title('Sobel Noreste')

axs[1][0].imshow(img5, cmap="gray")
axs[1][0].axis("off")
axs[1][0].set_title('Sobel Este')

axs[1][1].imshow(img6, cmap="gray")
axs[1][1].axis("off")
axs[1][1].set_title('Sobel Sureste')

axs[1][2].imshow(img7, cmap="gray")
axs[1][2].axis("off")
axs[1][2].set_title('Sobel Sur')

axs[2][0].imshow(img8, cmap="gray")
axs[2][0].axis("off")
axs[2][0].set_title('Sobel Suroeste')

axs[2][1].imshow(img9, cmap="gray")
axs[2][1].axis("off")
axs[2][1].set_title('Sobel Oeste')

axs[2][2].imshow(img10, cmap="gray")
axs[2][2].axis("off")
axs[2][2].set_title('Sobel Noroeste')










plt.show()
