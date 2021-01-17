import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import pickle
import os
import time


def load_model(file='model(1).pickle'):
    with open(file, 'rb') as f:
        model = pickle.load(f)

    return model.coef_[0]


def remove_green(img, coef, bg):

    newImg = img.copy()
    if img.shape != bg.shape:
        bg = cv.resize(bg, img.shape[-2::-1])

    w3, w2, w1, w0 = coef

    B, G, R = cv.split(img)
    inter = np.ones(R.shape)

    mask = w3 * B + w2 * G + w1 * R + w0 * inter

    mask = 1 / (1 + 1/np.exp(mask))

    threhold = 0.9999


    newImg[mask > threhold] = [0, 0, 0]
    bg[mask < threhold] = [0, 0, 0]

    return  newImg + bg

    # shape = img.shape
    # pixcels = img.copy().reshape(-1, 3)
    # for i in range(pixcels.shape[0]):
    #     features = np.append(pixcels[i].reshape(1, -1), [1]).reshape(1, -1)
    #     if model.predict(features)[0] == str(1):
    #         pixcels[i] = np.array([255, 255, 255])
    # newImg = pixcels.reshape(shape)


def process_video(path, bg, coef):

    cap = cv.VideoCapture(path)

    while (cap.isOpened()):
        ret, frame = cap.read()

        processed_frame = remove_green(frame, coef, bg)

        cv.imshow('frame', processed_frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()


def process(fg, bg, type_fg):

    coef = load_model()

    bg = cv.imread(bg)
    bg = cv.cvtColor(bg, cv.COLOR_BGR2RGB)

    if type_fg == 'video':

        pass
    else:
        fg = cv.imread(fg)
        fg = cv.cvtColor(fg, cv.COLOR_BGR2RGB)
        return remove_green(fg, coef, bg)


if __name__ == '__main__':

    coef = load_model()

    path = os.getcwd()
    nameImg = '1.jpg'
    pathImg = os.path.join(path, 'test_data', nameImg)

    pathImg_1 = os.path.join(path, 'back_ground', '2.jpg')

    img = cv.imread(pathImg, 1)
    bg = cv.imread(pathImg_1, 1)


    newImg = remove_green(img, coef, bg)

    cv.imshow('mask', newImg)
    cv.waitKey(0)
    cv.destroyWindow()
    # plt.subplot(1, 2, 1);
    # plt.imshow(img)
    #
    # plt.subplot(1, 2, 2);
    # plt.imshow(newImg)
    #
    # plt.show()
    #

    # path_video = os.path.join(path, 'video', 'helicopter-2.mp4')
    # process_video(path_video, bg, coef)