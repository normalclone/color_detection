import os
import cv2
import numpy as np


def color_histogram_of_test_image(test_src_image, test_file_path):
    # load the image
    image = test_src_image

    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)

        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue

    with open(test_file_path, 'w') as myfile:
        myfile.write(feature_data)



def color_histogram_of_training_image(img_name, data_source, feature_saved_path):
    # load the image
    image = cv2.imread(img_name)

    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)

        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue

    with open(feature_saved_path, 'a') as myfile:
        myfile.write(feature_data + ',' + data_source + '\n')


def training(data_path, feature_saved_path, override=False):
    if override:
        if os.path.isfile(feature_saved_path):
            os.remove(feature_saved_path)
    for d in os.listdir(data_path):
        for f in os.listdir(data_path + "/" + d):
            color_histogram_of_training_image(data_path + "/" + d + "/" + f, d, feature_saved_path)
