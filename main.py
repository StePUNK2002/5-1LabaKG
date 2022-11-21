import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def Action(way, g11, g22, fmin1, fmax1):
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

        R = (R * 0.299)
        G = (G * 0.587)
        B = (B * 0.114)

        Avg = (R + G + B)
        R = Avg
        G = Avg
        B = Avg

        Image = img.copy()

        for i in range(0, len(R)):
            for j in range(0, len(R[0])):
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

    image = mpimg.imread(way)
    grayImage = rgb_to_gray(image)
    image3 = TaskImg(grayImage, g11, g22, fmin1, fmax1)
    print(type(grayImage))
    fig, axs = plt.subplots(nrows=2, ncols=2)
    fig.suptitle('Фотографии')
    axs[0][0].imshow(image)
    axs[0][1].imshow(grayImage)
    axs[1][0].imshow(image3)
    axs[1][1].imshow(image3)
    plt.show()
#Action("berserk.jpeg", 255, 255, 1, 2)
