from django.views.decorators.csrf import csrf_exempt
import os
from django.http import HttpResponse
import cv2
import numpy as np
import json
import urllib.request
from .Utils.color_detection import color_detection
import uuid
import base64

data_path = "ColorDetection/training_dataset";
detector = color_detection(PATH='./training.data', split_number=50, data_path=data_path)

@csrf_exempt
def detect(request):
    try:
        response = HttpResponse()
        if request.method == 'GET':
            try:
                image = cv2.imread(request.GET["uri"])
                rs = detector.detect(image)
            except:
                req = urllib.request.urlopen(request.GET["uri"])
                arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
                image = cv2.imdecode(arr, -1)
                rs = detector.detect(image)
        else:
            image = cv2.imdecode(np.fromstring(request.FILES['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
            rs = detector.detect(image)

        response.writelines(rs)
        return response
    except Exception as e:
        return HttpResponse(repr(e))

@csrf_exempt
def train(request):
    try:
        response = HttpResponse()
        detector.train()
        rs = "Success"
        response.writelines(rs)
        return response
    except Exception as e:
        return HttpResponse(repr(e))

@csrf_exempt
def insert_data(request):
    try:
        response = HttpResponse()
        if request.method == 'GET':
            try:
                image = cv2.imread(request.GET["uri"])
            except:
                req = urllib.request.urlopen(request.GET["uri"])
                arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
                image = cv2.imdecode(arr, -1)

            color = str(request.GET["color"])
            if not os.path.exists(data_path+"/"+color):
                os.mkdir(data_path+"/"+color)

            cv2.imwrite(data_path+"/"+color+"/"+str(uuid.uuid4())+".jpg", image)

        else:
            image = cv2.imdecode(np.fromstring(request.FILES['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
            color = str(request.POST["color"])
            if not os.path.exists(data_path+"/"+color):
                os.mkdir(data_path+"/"+color)
            cv2.imwrite(data_path+"/"+color+"/"+str(uuid.uuid4())+".jpg", image)
        rs = "Success! Access train API to re-train model"
        response.writelines(rs)
        return response
    except Exception as e:
        return HttpResponse(repr(e))

@csrf_exempt
def get_list_color(request):
    try:
        response = HttpResponse()
        response.writelines(", ".join(os.listdir(path=data_path)))
        return response
    except Exception as e:
        return HttpResponse(repr(e))