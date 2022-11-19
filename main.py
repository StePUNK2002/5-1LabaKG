import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def Action(way):
    def rgb_to_gray(img):
        grayImage = np.zeros(img.shape)
        #срезы
        R = np.array(img[:, :, 0])
        G = np.array(img[:, :, 1])
        B = np.array(img[:, :, 2])
        print(R)

        R = (R * 0.299)
        G = (G * 0.587)
        B = (B * 0.114)

        Avg = (R + G + B)
        grayImage = img.copy()

        for i in range(3):
            grayImage[:, :, i] = Avg

        return grayImage

    image = mpimg.imread(way)
    grayImage = rgb_to_gray(image)
    print(type(grayImage))
    fig, axs = plt.subplots(nrows=1, ncols=2)
    fig.suptitle('Фотографии')
    axs[0].set_title('Оригинальное изображение')
    axs[0].imshow(image)
    axs[1].set_title('Оттенки серого')
    axs[1].imshow(grayImage)
    plt.show()

