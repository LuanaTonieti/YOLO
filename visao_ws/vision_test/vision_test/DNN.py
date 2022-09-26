import argparse
import time
import cv2
import pandas as pd

# import the necessary packages
import imutils

import os
#import tarfile
import time
import os
import cv2
import numpy as np
import sys

#from servo import Servo

import random as rd
import tensorflow as tf

from telnetlib import NOP
import rclpy
from rclpy.node import Node

#from std_msgs.msg import String
from custom_interfaces.msg import Vision


from .models import *

import sys
sys.path.insert(0, './vision_test')
import numpy as np
import cv2
import ctypes
from math import log,exp,tan,radians
from .camvideostream import WebcamVideoStream
#import imutils

from .serialization import *

try:
    """There are differences in versions of the config parser
    For versions > 3.0 """
    from configparser import ConfigParser
except ImportError:
    """For versions < 3.0 """
    from configparser import ConfigParser 

import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random


from .experimental import attempt_load
from .utils.datasets import LoadStreams, LoadImages
from .utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from .utils.plots import plot_one_box
from .utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

PATH_TO_WEIGHTS = '/home/robofei/Desktop/visao_ws/vision_test/vision_test/best.pt'

#SERVO_PAN = 19
#SERVO_TILT = 20

#SERVO_TILT_VALUE = 705 # Posicao central inicial Tilt
#SERVO_PAN_VALUE = 512 # Posicao central inicial Tilt




class objectDetect():
    CountLostFrame = 0
    Count = 0
    status =1
    statusLost = 0

    def __init__(self):

        NUM_CLASSES = 1


    def searchball(self, image): 
        source = '2'
        weights = PATH_TO_WEIGHTS
        view_img = True
        imgsz = 640
        trace = False
        img_size = 640
        augment = True
        conf_thres = 0.25
        iou_thres = 0.45
        classes = 0
        agnostic_nms = True

        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

        set_logging()
        device = select_device('')
        half = False

     # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size

        if trace:
            model = TracedModel(model, device, img_size)

        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1

        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Warmup
            if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    model(img, augment=augment)[0]

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes=classes, agnostic=agnostic_nms)
            t3 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    print(det)

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        if n>0:
                            BallFound = True

                # Print time (inference + NMS)
                print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

                # Stream results
                if view_img:
                    cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

        print(f'Done. ({time.time() - t0:.3f}s)')
			
        return im0, det[0], det[2], det[1]-det[0], BallFound, self.status

    #Varredura
   # def SearchLostBall(self):

        #if self.bkb.read_int(self.Mem,'IMU_STATE')==0:
            
            #if self.Count == 0:
            #    self.servo.writeWord(self.config.SERVO_PAN_ID,self.servo.ADDR_PRO_GOAL_POSITION  , self.config.CENTER_SERVO_PAN - self.config.SERVO_PAN_LEFT) #olha para a esquerda
            #    time.sleep(1)
            #    self.Count +=1
            #    return 0
            #if self.Count == 1:
            #    self.servo.writeWord(self.config.SERVO_PAN_ID,self.servo.ADDR_PRO_GOAL_POSITION , self.config.CENTER_SERVO_PAN)#olha para o centro
            #    time.sleep(1)
            #    self.Count +=1
            #    return 1
            #if self.Count == 2:
            #    self.servo.writeWord(self.config.SERVO_PAN_ID,self.servo.ADDR_PRO_GOAL_POSITION , self.config.CENTER_SERVO_PAN + self.config.SERVO_PAN_RIGHT)#olha para a direita 850- 440
            #    time.sleep(1)
            #    self.Count +=1
            #    return 2
            #if self.Count == 3:
            #    self.servo.writeWord(self.config.SERVO_PAN_ID,self.servo.ADDR_PRO_GOAL_POSITION , self.config.CENTER_SERVO_PAN)#olha pro centro
            #    time.sleep(1)
            #    self.Count = 0
            #    return 1


    def Morphology(self, frame):

        start3 = time.time()
        contador = 0
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image_np = np.asarray(frame)

      # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        (boxes, scores, classes, num) = self.__sess.run(
            [self.__detectionboxes, self.__detectionscores, self.__detectionclasses, self.__numdetections],
            feed_dict={self.__imagetensor: image_np_expanded})
      # Visualization of the results of a detection.

        vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          self.category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
#      plt.figure(figsize=IMAGE_SIZE)
#      plt.imshow(image_np)

        df = pd.DataFrame()
        df['classes'] = classes[0]
        df['scores'] = scores[0]
        df['boxes'] = boxes[0].tolist()

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if(df['scores'][0]>0.60):
            height, width = frame.shape[:2]
            print(df['boxes'][0][0])
            #      print df.head()

            #      box_coords[ymin, xmin, ymax, xmax]
            y1 = int(df['boxes'][0][0]*height)
            x1 = int(df['boxes'][0][1]*width)
            y2 = int(df['boxes'][0][2]*height)
            x2 = int(df['boxes'][0][3]*width)
            xmed = (x2-x1)/2
            ymed = (y2-y1)/2
            return frame, x2-xmed, y2-ymed, (xmed+ymed)/2

        #=================================================================================================
        return frame, 0, 0, 0





