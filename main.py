import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def Action(way, g11, g22, fmin1, fmax1, mat=[[-1, 0, 0],[0,0,0],[0,0,1]]):
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

        # Матрица определения границ по второй производной функции Гаусса

        mat = [[-1, 0, 0], [0, 0, 0], [0, 0, 1]]
        #        mat = [[-1,0,0], [0,0,0], [0,0,1]]
        R = np.array(img[:, :, 0])
        G = np.array(img[:, :, 1])
        B = np.array(img[:, :, 2])

        Avg = R + G + B
        Res = R + G + B
        Tem = R + G + B

        Avg = np.resize(Avg, (len(R) + 2, len(R[0]) + 2))
        Res = np.resize(Res, (len(R) + 2, len(R[0]) + 2))

        for i in range(0, len(Tem) - 1):
            for j in range(0, len(Tem[0] - 1)):
                Avg[i + 1][j + 1] = Tem[i][j]
                if i == 0:
                    Avg[i][j + 1] = Tem[i][j]
                if j == 0:
                    Avg[i + 1][j] = Tem[i][j]
                if i == len(Tem) - 1:
                    Avg[i + 2][j + 1] = Tem[i][j]
                if j == len(Tem[0]) - 1:
                    Avg[i + 1][j + 2] = Tem[i][j]

        for i in range(1, len(Avg) - 2):
            for j in range(1, len(Avg[0]) - 2):
                Res[i][j] = AM + BM * (
                            Avg[i - 1][j - 1] * mat[0][0] + Avg[i - 1][j] * mat[1][0] + Avg[i - 1][j + 1] * mat[2][0] +
                            Avg[i][j - 1] * mat[0][1] + Avg[i][j] * mat[1][1] + Avg[i][j + 1] * mat[2][1] + Avg[i + 1][
                                j - 1] * mat[0][2] + Avg[i + 1][j] * mat[1][2] + Avg[i + 1][j + 1] * mat[2][2])

                if Res[i][j] > 255:
                    Res[i][j] = 255
                elif Res[i][j] < 0:
                    Res[i][j] = 0

        for i in range(0, len(Tem) - 1):
            for j in range(0, len(Tem[0]) - 1):
                Tem[i][j] = Res[i + 1][j + 1]
        Res = Tem

        Image = np.zeros(img.shape)
        Image = img.copy()
        Image[:, :, 0] = Res
        Image[:, :, 1] = Res
        Image[:, :, 2] = Res
        return Image

    image = mpimg.imread(way)
    grayImage = rgb_to_gray(image)
    image3 = TaskImg(grayImage, g11, g22, fmin1, fmax1)
    image4 = Task3(grayImage,mat)
    fig, axs = plt.subplots(nrows=2, ncols=2)
    fig.suptitle('Фотографии')
    axs[0][0].imshow(image)
    axs[0][1].imshow(grayImage)
    axs[1][0].imshow(image3)
    axs[1][1].imshow(image4)
    plt.show()
Action("berserk.jpeg", 0, 255, 40, 60)
