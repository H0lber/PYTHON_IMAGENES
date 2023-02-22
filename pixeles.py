import cv2
from matplotlib import pyplot as plt

img = cv2.imread('si2.jpg',0)
img2 = img.copy()

(row, col) = img.shape

for i in range(row):
    for j in range(col):
        if (img[i][j] < 10):
            img2[i][j] = 10
        if (img[i][j] > 240):
            img2[i][j] = 240

fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize =(8,8))
axs[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs[0].axis("off")

axs[1].imshow(img2, cmap="Greys")
axs[1].axis("off")

plt.show()
