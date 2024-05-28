import numpy as np

from matplotlib import pyplot as plt
from math import sqrt
import cv2 as cv

if __name__ == "__main__":


    #Ouvrir une image
    img = cv.imread("CerisierP.jpg")
    img = cv.cvtColor(img,cv.COLOR_BGR2RGB)

    #Afficher une image
    #plt.figure()
    #plt.imshow(img)

    #Afficher les canaux
    rouge = img[:,:, 0]
    vert = img[:,:, 1]
    bleu = img[:,:, 2]
    #plt.figure()
    #plt.subplot(2, 2, 1)
    #plt.imshow(img)
    #plt.subplot(2, 2, 2)
    #plt.imshow(rouge, cmap="Reds")
    #plt.subplot(2, 2, 3)
    #plt.imshow(vert, cmap="Greens")
    #plt.subplot(2, 2, 4)
    #plt.imshow(bleu, cmap="Blues")

    # Transformation en niveau de gris
    imgG = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
    plt.figure()
    plt.imshow(imgG, cmap="gray")

   # plt.show()
    
    plt.figure()
    plt.hist(imgG)
    
    def binariser(seuil, imageG):
        imageGbis = imageG[::]
        for i in range(0,len(imageG)):
            for j in range(0,len(imageG[0])):
                if (imageG[i][j] < seuil):
                    imageGbis[i][j] = 0
                else:
                    imageGbis[i][j] = 1
        return imageGbis



    def calculSeuil(imageG):
        histo, bins = np.histogram(imageG, bins=256, range=(0, 256))
        nbTot = imageG.size
        m1, m2, m3 = 0, 0, 0
        #Calcul des différents moments
        m0 = np.sum(histo)
        m1 = np.sum(np.arange(256) * histo) / nbTot
        m2 = np.sum((np.arange(256)**2) * histo) / nbTot
        m3 = np.sum((np.arange(256)**3) * histo) / nbTot        
        #Calcul de C0 et C1
        C1 = (m1*m2 - m3)/(m2-m1**2)
        C0 = -(m1*C1 + m2)
        #calcul de z0 et z1
        z0 = (-C1 + sqrt(C1**2 - 4*C0))/2
        z1 = (-C1 - sqrt(C1**2 - 4*C0))/2
        #calcul du seuil
        seuil = int((z0+z1)/2)
        print (seuil)
        return(seuil)

    plt.figure()
    plt.imshow(binariser(calculSeuil(imgG), imgG), cmap = "gray")

                
    
    
    
    plt.show()
    
