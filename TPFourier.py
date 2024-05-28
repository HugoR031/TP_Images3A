import numpy as np

from matplotlib import pyplot as plt
import cv2 as cv
from mpl_toolkits import mplot3d

def affiche_image(img):
    plt.figure()
    plt.imshow(img,cmap="gray")
    plt.colorbar()

def  atom(n,m,fx,fy):
    img=np.zeros((n, m))
    x = np.array(np.arange(0,m))
    y = np.arange(0,n)
    e1 = np.exp(1j*2*np.pi*fx*x)
    e2 = np.exp(1j*2*np.pi*fy*y)
    for i in range(n):
        for j in range(m):
            img[i,j] = np.real(e2[i]*np.conjugate(e1[j]))
    return img

def fourier2d(img,fe):
    #Récupération de la hauteur et de la largeur de l'image
    [height, width] =img.shape
    #Calcul de la transformée de Fourier de l'image
    f = np.abs(np.fft.fftshift(np.fft.fft2(img))) 
    n = width/2
    m = height/2
    #Création d'une figure en 3D 
    plt.figure()
    ax = plt.axes(projection='3d')
    x = np.arange(-n/width, n/width, float(fe/width))
    y = np.arange(-m/height, m/height, float(fe/height))
    X, Y = np.meshgrid(x, -y)
    #Affichage du spectre 
    print(X.shape)
    ax.plot_surface(X, Y, np.sqrt(f))
    plt.title({"Spectre - 1"})
    plt.xlabel("Fx")
    plt.ylabel("Fy")
    #Affichage du spectre avec une échelle logarithmique
    plt.figure()
    plt.imshow(np.log(5*f+1),extent=[-n/width, n/width, -m/height, m/height])
    plt.colorbar()
    plt.xlabel("Fx")
    plt.ylabel("Fy")
    plt.title("Spectre - 2")


if __name__ == "__main__":


    img = atom(128,128,0.1,0)
    affiche_image(img)    
    fourier2d(img,1)
    
    img2 = atom(128,128,0,0.1)
    affiche_image(img2)
    fourier2d(img2,1)
    
    img3 = atom(128,128,0.3,0.3)
    affiche_image(img3)
    fourier2d(img3,1)



    plt.show()



