import numpy as np
from PIL import ImageFilter, Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def Action(way, g11, g22, fmin1, fmax1, mat=[[-1, 0, 0],[0,1,0],[0,0,0]]):
    def rgb_to_gray(img):
        grayImage = np.zeros(img.shape)
        #срезы
        R = np.array(img[:, :, 0])
        G = np.array(img[:, :, 1])
        B = np.array(img[:, :, 2])

        R = (R * 0.299)
        G = (G * 0.587)
        B = (B * 0.114)

        Avg = (R + G + B)
        grayImage = img.copy()

        for i in range(3):
            grayImage[:, :, i] = Avg

        return grayImage

    def TaskImg(img, g1, g2, fmin, fmax):
        Image = np.zeros(img.shape)
        R = np.array(img[:, :, 0])
        G = np.array(img[:, :, 1])
        B = np.array(img[:, :, 2])

        Avg = (R + G + B)

        Image = img.copy()

        for i in range(0, len(R)): #высота
            for j in range(0, len(R[0])): #ширина
                if Avg[i][j] < fmin:
                    Avg[i][j] = g1
                if Avg[i][j] > fmax:
                    Avg[i][j] = g2
                if Avg[i][j] > fmin and Avg[i][j] < fmax:
                    Avg[i][j] = (Avg[i][j] - g1)*(255 - 0) / ((g2 - g1) + 0)

        Image[:, :, 0] = Avg
        Image[:, :, 1] = Avg
        Image[:, :, 2] = Avg

        return Image

    def Task3(img, mat):
        AM = 128
        BM = 1 / 2

        R = np.array(img[:, :, 0])
        G = np.array(img[:, :, 1])
        B = np.array(img[:, :, 2])


        Avg = (R + G + B)
        Res = (R + G + B)

        for i in range(1, len(R) - 2):
            for j in range(1, len(R[0]) - 2):
                temp = 0
                for im in range(-1, 1):
                    for jm in range(-1, 1):
                        temp = temp + Avg[i + im][j + jm] * mat[im + 1][jm + 1]
                Res[i][j] = AM + BM * temp
        H = len(R)-1
        W = len(R[0])-1
        Res[0][0] = AM + BM * (Avg[0][0] * mat[1][1] + Avg[0][1] * mat[1][2] + Avg[1][0] * mat[2][1] + Avg[1][1] * mat[2][2])
        Res[H][0] = AM + BM * (Avg[H][0] * mat[1][1] + Avg[H][1] * mat[1][2] + Avg[H - 1][0] * mat[0][0] + Avg[H - 1][1] * mat[0][1])
        Res[0][W] = AM + BM * (Avg[0][W] * mat[1][1] + Avg[0][W - 1] * mat[1][0] + Avg[1][W] * mat[2][1] + Avg[1][W - 1] * mat[0][2])
        Res[H][W] = AM + BM * ( Avg[H][W] * mat[1][1] + Avg[H - 1][W] * mat[0][1] + Avg[H][W - 1] * mat[1][0] + Avg[H - 1][W - 1] *mat[0][0])
        Image = np.zeros(img.shape)
        Image = img.copy()
        Image[:, :, 0] = Res
        Image[:, :, 1] = Res
        Image[:, :, 2] = Res
        return Image

    image = mpimg.imread(way)
    grayImage = rgb_to_gray(image)
    image3 = TaskImg(grayImage, g11, g22, fmin1, fmax1)
    image4 = Task3(image,mat)
    fig, axs = plt.subplots(nrows=2, ncols=2)
    fig.suptitle('Фотографии')
    axs[0][0].imshow(image)
    axs[0][1].imshow(grayImage)
    axs[1][0].imshow(image3)
    axs[1][1].imshow(image4)
    plt.show()
Action("berserk.jpeg", 255, 255, 1, 2)
