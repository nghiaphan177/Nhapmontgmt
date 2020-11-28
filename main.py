import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import pickle
import os


def load_model(file='model.pickle'):
    with open(file, 'rb') as f:
        model = pickle.load(f)

    return model


def remove_green(img, model):
    shape = img.shape
    pixcels = img.copy().reshape(-1, 3)
    for i in range(pixcels.shape[0]):
        features = np.append(pixcels[i].reshape(1, -1), [1]).reshape(1, -1)
        if model.predict(features)[0] == str(1):
            pixcels[i] = np.array([255, 255, 255])
    newImg = pixcels.reshape(shape)
    return newImg


if __name__ == '__main__':
    path = os.getcwd()
    nameImg = '2.jpg'
    pathImg = os.path.join(path, 'test_data', nameImg)
    img = cv.imread(pathImg, 1)

    model = load_model()
    newImg = remove_green(img, model)
    plt.subplot(1, 2, 1);
    plt.imshow(img)

    plt.subplot(1, 2, 2);
    plt.imshow(newImg)

    plt.show()
