import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import pickle
import os
import time


def load_model(file='model.pickle'):
    with open(file, 'rb') as f:
        model = pickle.load(f)

    return model


def remove_green(img, model):

    newImg = img.copy()

    w3, w2, w1, w0 = model.coef_[0]

    R = img[:, :, 0]
    G = img[:, :, 1]
    B = img[:, :, 2]
    inter = np.ones(R.shape)

    mask = w3 * R + w2 * G + w1 * B + w0 * inter

    mask = 1 / (1 + 1/np.exp(mask))

    newImg[mask > 0.99999999] = [0, 0, 0]


    # shape = img.shape
    # pixcels = img.copy().reshape(-1, 3)
    # for i in range(pixcels.shape[0]):
    #     features = np.append(pixcels[i].reshape(1, -1), [1]).reshape(1, -1)
    #     if model.predict(features)[0] == str(1):
    #         pixcels[i] = np.array([255, 255, 255])
    # newImg = pixcels.reshape(shape)
    return newImg


if __name__ == '__main__':
    path = os.getcwd()
    nameImg = '2.jpg'
    pathImg = os.path.join(path, 'test_data', nameImg)
    img = cv.imread(pathImg, 1)

    start = time.time()
    model = load_model()
    print(time.time() - start)

    newImg = remove_green(img, model)
    plt.subplot(1, 2, 1);
    plt.imshow(img)

    plt.subplot(1, 2, 2);
    plt.imshow(newImg)

    plt.show()
