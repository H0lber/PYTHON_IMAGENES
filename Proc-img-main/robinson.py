import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('sponge.jpg', 0)
img2 = img.copy()

(row, col) = img.shape

for i in range(row):
    for j in range(col):
        if (img[i][j] < 10):
            img2[i][j] = 10
        if (img[i][j] > 240):
            img2[i][j] = 240

f_max = img2.max()
f_min = img2.min()
img3 = img2.copy()

for i in range(row):
    for j in range(col):
        img3[i][j] = ((img2[i][j] - f_min) / (f_max - f_min)) * 256

edge_kernel = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
norte = cv2.filter2D(img3,-1,edge_kernel)

edge_kernel = np.array([[-2,-1,0],[-1,0,1],[0,1,2]])
noroeste = cv2.filter2D(img3,-1,edge_kernel)

edge_kernel = np.array([[-1,-2,-1],[0,0,0],[1,2,1]])
este = cv2.filter2D(img3,-1,edge_kernel)

edge_kernel = np.array([[0,-1,-2],[1,0,-1],[2,1,0]])
sureste = cv2.filter2D(img3,-1,edge_kernel)

edge_kernel = np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
sur = cv2.filter2D(img3,-1,edge_kernel)

edge_kernel = np.array([[2,1,0],[1,0,-1],[0,-1,-2]])
suroeste = cv2.filter2D(img3,-1,edge_kernel)

edge_kernel = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
oeste = cv2.filter2D(img3,-1,edge_kernel)

edge_kernel = np.array([[0,1,2],[-1,0,1],[-2,-1,0]])
noreste = cv2.filter2D(img3,-1,edge_kernel)


fig, axs = plt.subplots(nrows=3, ncols=4, figsize=(8, 8))

axs[0][0].imshow(norte, cmap="gray")
axs[0][0].axis("off")
axs[0][0].set_title('Norte')


axs[1][0].imshow(noroeste, cmap="gray")
axs[1][0].axis("off")
axs[1][0].set_title('NorOeste')

axs[0][1].imshow(este, cmap="gray")
axs[0][1].axis("off")
axs[0][1].set_title('Este')

axs[1][1].imshow(sureste, cmap="gray")
axs[1][1].axis("off")
axs[1][1].set_title('SurEste')

axs[0][2].imshow(sur, cmap="gray")
axs[0][2].axis("off")
axs[0][2].set_title('Sur')

axs[1][2].imshow(sur, cmap="gray")
axs[1][2].axis("off")
axs[1][2].set_title('Sur')

axs[0][3].imshow(oeste, cmap="gray")
axs[0][3].axis("off")
axs[0][3].set_title('Oeste')

axs[1][3].imshow(noreste, cmap="gray")
axs[1][3].axis("off")
axs[1][3].set_title('NorOeste')

axs[2][0].axis("off")

axs[2][1].imshow(img, cmap="gray")
axs[2][1].axis("off")
axs[2][1].set_title('Original')

axs[2][2].axis("off")
axs[2][3].axis("off")


plt.show()
cv2.destroyAllWindows()