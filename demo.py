
import numpy as np

from matplotlib import pyplot as plt
import cv2 as cv

if __name__ == "__main__":


    #Ouvrir une image
    img = cv.imread("CerisierP.jpg")
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)

    #Afficher une image
    plt.figure()
    plt.imshow(img)

    #Afficher les canaux
    rouge = img[:,:, 0]
    vert = img[:,:, 1]
    bleu = img[:,:, 2]
    plt.figure()
    plt.subplot(2, 2, 1)
    plt.imshow(img)
    plt.subplot(2, 2, 2)
    plt.imshow(rouge, cmap="Reds")
    plt.subplot(2, 2, 3)
    plt.imshow(vert, cmap="Greens")
    plt.subplot(2, 2, 4)
    plt.imshow(bleu, cmap="Blues")

    # Transformation en niveau de gris
    imgG = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    plt.figure()
    plt.imshow(imgG, cmap="gray")

    plt.show()
