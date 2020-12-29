from ..Core import color_histogram_feature_extraction
from ..Core import knn_classifier
import csv
import os
import uuid
import numpy as np

class color_detection:
    def __init__(self, PATH='./training.data', data_path="ColorDetection/training_dataset", split_number=50):
        self.PATH = PATH
        self.split_number = split_number
        self.data_path = data_path
        self.training_feature_vector = []
        self.loadDataset(PATH)

    def loadDataset(self, filename):
        self.training_feature_vector = []
        try:
            with open(filename) as csvfile:
                lines = csv.reader(csvfile)
                dataset = list(lines)
                for x in range(len(dataset)):
                    for y in range(3):
                        dataset[x][y] = float(dataset[x][y])
                    self.training_feature_vector.append(dataset[x])
        except:
            self.train()

    def train(self):
        color_histogram_feature_extraction.training(data_path=self.data_path, override=True, feature_saved_path=self.PATH)
        self.loadDataset(self.PATH)


    def detect(self, image):
        height, width = image.shape[:2]
        splitedFrames = []
        sub_frame_height = int(height / self.split_number)
        sub_frame_width = int(width / self.split_number)
        while len(splitedFrames) < self.split_number:
            splitedFrames.append(
                image[
                (sub_frame_height * len(splitedFrames)):(sub_frame_height * len(splitedFrames)) + sub_frame_height,
                (sub_frame_width * len(splitedFrames)):(sub_frame_width * len(splitedFrames)) + sub_frame_width],
            )
        rs = []
        for i in splitedFrames:
            testp = str(uuid.uuid4())
            color_histogram_feature_extraction.color_histogram_of_test_image(i, testp)
            prediction = knn_classifier.main(testp, self.training_feature_vector)
            rs.append(prediction)
            os.remove(testp)
        scoring = []
        for i in rs:
            try:
                elems = [j for j, x in enumerate(scoring) if x[0] == i]
                if len(elems) > 0:
                    if i == "black":
                        scoring[elems[0]][1] = scoring[elems[0]][1] + 0.5
                    else:
                        scoring[elems[0]][1] = scoring[elems[0]][1] + 1
                else:
                    scoring.append([i, 0])
            except:
                scoring.append([i, 0])
        max = scoring[0]
        for i in scoring:
            if i[1] >= max[1]:
                max = i

        return max[0]
