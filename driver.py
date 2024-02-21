import numpy as np
import mediapipe as mp
import logging
import sys
import cv2
import glob
import matplotlib.pyplot as plt
from tensorflow.keras.utils import img_to_array
from keras.models import load_model
from statistics import mean
from sklearn.cluster import KMeans
from collections import Counter
import imutils
from skimage import feature, img_as_ubyte
import yaml
import os
from pathlib import Path
import warnings
from deepface import DeepFace
# import argparse
# import platform
# import torch
import torch.backends.cudnn as cudnn
from ultralytics import YOLO
from uuid import uuid4
from dominant_color_detection import detect_colors
from PIL import ImageColor
from numpy import random
from PIL import Image, ImageDraw, ImageFilter
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import base64

warnings.filterwarnings('ignore')

# Create and configure logger
fmt = '%(asctime)s : %(filename)s : %(lineno)d : %(funcName)s : %(message)s'

info = logging.FileHandler("logsFile/Info.log", mode='a')
info.setLevel(logging.INFO)

error = logging.FileHandler("logsFile/Error.log", mode='a')
error.setLevel(logging.ERROR)

console_handler = logging.StreamHandler()  
console_handler.setLevel(logging.ERROR)

logging.basicConfig(level=logging.INFO, format=fmt, handlers=[info, error, console_handler])

# Creating an object
logger = logging.getLogger()
logger.info("-----------------------------------------------------------------------------------------")


#----------------------- Models Path -------------------------
wrinkle_models_path = ["models/w_old.pt","models/w_new.pt"]
acne_model_path_new = "models/acne_spot_new.pt"
acne_model_path_old = "models/acne_spot_old.pt"
age_model_path = "models/age_exp91.pt"
#-------------------------------------------------------------


'''
Face Extracted used for detection of face from the FULL face image.
This is done with Mediapipe Library
'''
def face_extractor(img):
    try:
        mp_face_detection = mp.solutions.face_detection
        face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils
        h,w,_ = img.shape 
        face_detection_results = face_detection.process(img[:,:,::-1])
        if face_detection_results.detections:
            for face_no, face in enumerate(face_detection_results.detections):
                face_data = face.location_data
                faces =[[int(face_data.relative_bounding_box.xmin*w),int(face_data.relative_bounding_box.ymin*h),
                         int(face_data.relative_bounding_box.width*w),int(face_data.relative_bounding_box.height*h)]]
                for x, y, w, h in faces:
                    cropped_img = img[y-int(h/3):y + 13*int(h/12), x:x + w]
        if cropped_img is None:
            print("Could not read input image")
            sys.exit()
        logger.info("OK")
    except Exception as e:
        logger.error(f"Exception occurred in Face Extractor {e}", exc_info=True)
        cropped_img = False
    return cropped_img

def face_existence(face):    
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.3)
    mp_drawing = mp.solutions.drawing_utils
    face_detection_results = face_detection.process(face[:,:,::-1])
    if face_detection_results.detections:
        for face_no, face in enumerate(face_detection_results.detections):
            if mp_face_detection.FaceKeyPoint(1).name == 'LEFT_EYE' or mp_face_detection.FaceKeyPoint(0).name == 'RIGHT_EYE':
                return True
    else:
        return False
   
'''
This function is used to mask(show only inside coordinated area and fill black outside the ROI) the face image
'''    
def face_masking(face_cropped_img,list_landmarks):
    try:
        face = face_cropped_img.copy()
        mask = np.zeros((face_cropped_img.shape[0],face_cropped_img.shape[1]))
        cv2.drawContours(mask, np.array(list_landmarks), -1, 255, -1)
        mask = mask.astype(np.bool_)
        masked_face = np.zeros_like(face)
        masked_face[mask] = face[mask]
        logger.info("OK")
    except Exception as e:
        logger.error(f"Exception occurred in face_masking  {e}", exc_info=True)
        masked_face = False
    return masked_face


# --------------------------------------- Wrinkle Model ------------------------------------------------------
def landmark_wrinkle(img):
    with open('mediapipe_roi//roi.yml') as f:
        config = list(yaml.safe_load_all(f))
    variable = config[0]
    wrinkle_roi = variable['wrinkle']
    points_wrinkle = wrinkle_roi['full_face']['landmarks']
    list_landmarks_wrinkle = []
    h,w,_ = img.shape
    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2,
                                            min_detection_confidence=0.5)
        face_mesh_results = face_mesh_images.process(img[:,:,::-1])
        if face_mesh_results.multi_face_landmarks:
            for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
                for ind in points_wrinkle:
                    list_landmark=[]
                    for pnt in ind:
                        list_landmark.append([int(face_landmarks.landmark[pnt].x*w),
                                              int(face_landmarks.landmark[pnt].y*h)])
                    list_landmarks_wrinkle.append(list_landmark)
        logger.info("OK")
    except Exception as e:
        print(e)
        list_landmarks_wrinkle = False
        logger.error(f"Exception occurred in landmark wrinkle {e}", exc_info=True)
    return list_landmarks_wrinkle


def mask_wrinkle(face):
    lw= landmark_wrinkle(face)
    full_face_lm = [lw[-1]]
    rest_lm = lw[:-1]
    h,w,_ = face.shape
    masked_face = np.zeros((h,w,3), np.uint8)
    cv2.drawContours(masked_face, np.array(full_face_lm), -1,(255,255,255), -1)
    white_image = np.ones((h,w,3), np.uint8)*255
    for j in rest_lm:
        contour_list = [j]
        contour_new = np.array(contour_list)
        cv2.drawContours(white_image, contour_new, -1, (0,0,0), -1)
    return masked_face,white_image

def wrinkle_image(img,wrinkle_model_path):
    try:
        h,w,_ = img.shape
        face = face_extractor(img)
        w_mask_img,rest_parts = mask_wrinkle(face)
        dir1 = 'face_extr'
        Path('face_extr').mkdir(parents=True, exist_ok=True)
        for file in os.scandir(dir1):
            os.remove(file.path)
        cv2.imwrite("face_extr/face.jpg",face)
        img_path = "face_extr/face.jpg"
        model = YOLO(wrinkle_model_path)
        face_copy = face.copy()
        flag= False
        print("wrinkle mask is being prepared")
        try:
            results = model.predict(source=img_path, conf=0.025,imgsz=800, classes=[0])
        except Exception as e:
            print(e)
            flag= False
            return face_copy, flag
        print("wrinkle Mask Prepared")
        if results[0].masks is None:
            flag=False
            return face_copy,flag
        else:
            tensor_obj = results[0].masks.masks
            arr = tensor_obj.cpu().detach().numpy()
            conf_list = list(results[0].boxes.conf.cpu().numpy())
            coord_list_for_puttext = [(x[0],x[1]) for x in results[0].boxes.boxes.cpu().numpy().astype("int")]
            del(results)
            del(tensor_obj)
            image = cv2.resize(face.copy(),(arr[0].shape[1],arr[0].shape[0]))
            h,w = image.shape[0:2]
            blank_image = np.zeros((h,w,3), np.uint8)
            for i in range(len(arr)):
                contours, hierarchy = cv2.findContours(np.array(arr[i], np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(blank_image, contours, -1, (255,255,255),-1)
                
            # BITWISE CODE FOR RESULTANT IMAGE
            he,we,_ =face.shape 
            blank_image = cv2.resize(blank_image,(we,he))
            dest_and1 = cv2.bitwise_and(blank_image, rest_parts, mask = None)  
            dest_and2 = cv2.bitwise_and(dest_and1,w_mask_img , mask = None)
            gray_image = cv2.cvtColor(dest_and2, cv2.COLOR_BGR2GRAY)
            ret, thresh = cv2.threshold(gray_image, 120, 255, cv2.THRESH_BINARY)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(face_copy, contours, -1, (20,200,10),-1)
            alpha = 0.7
            image_new_overlayed = cv2.addWeighted(face, alpha, face_copy, 1 - alpha, 0)
            logger.info("OK")
    except Exception as e:
        logger.error(f"Exception occurred in wrinkle_image {e}", exc_info=True)
        image_new_overlayed = False
    return image_new_overlayed

def driver_wrinkle(img):
    wrinkle_models_path = ["models/w_old.pt","models/w_new.pt"]
    w_list = []
    for path in wrinkle_models_path:
        w_img = wrinkle_image(img,path)
        w_list.append(w_img)
    return w_list[0],w_list[1]


# --------------------------------------------Acne----------------------------------------------------------- 

# Acne + Spots
def landmark_acne(face_cropped_img, landmarks):
    list_landmarks_acne = []
    h,w,_ = face_cropped_img.shape
    with open('mediapipe_roi//roi.yml') as f:
        config = list(yaml.safe_load_all(f))
    variable = config[0]
    acne = variable['acne']
    points_acne = acne['full_face']['landmarks']
    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh_images = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=2, min_detection_confidence=0.5)
        face_mesh_results = face_mesh_images.process(face_cropped_img[:,:,::-1])
        if face_mesh_results.multi_face_landmarks:
            for face_no, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
                for ind in points_acne:
                    list_landmark=[]
                    for pnt in ind:
                        list_landmark.append([int(face_landmarks.landmark[pnt].x*w),
                                              int(face_landmarks.landmark[pnt].y*h)])
                    list_landmarks_acne.append(list_landmark)
        list2 = list_landmarks_acne.copy()
        param1 = int(0.1 * h)
        param2 = int(0.05 * h)
        for i in range(len(list2[1])):   
            if i==0 or i==8: 
                list2[1][i] = [list2[1][i][0],list2[1][i][1]-param2]
            else:
                list2[1][i] = [list2[1][i][0],list2[1][i][1]-param1]
        for i in list2[1]:
            list2[0].append(i)
        full_face_coor = [list2[0]]
        logger.info("OK")
    except Exception as e:
        print(f"Exception occurred in landmark_acne {e}", exc_info=True)
        full_face_coor = False
    return full_face_coor


def landmarks_roi_selection(img):
    try:
        with open('mediapipe_roi//roi.yml') as f:
            config = list(yaml.safe_load_all(f))
        variable = config[0]
        acne = variable['acne']
        points_acne_full_face = acne['full_face']['landmarks']
        logger.info("OK")
    except Exception as e:
        logger.error(f"Exception occurred in landmarks_roi_selection {e}", exc_info=True)
        points_acne_full_face = "Points not available for ACNE"
    return points_acne_full_face

# Acne OLD
def detect_acne_old(img_path,face,my_tuple):
    try:
        model = YOLO(acne_model_path_old)
        results = model.predict(source=img_path, conf=0.02,iou=0.1,classes=[0,1,2],imgsz=640, agnostic_nms=True)
        image = cv2.imread(img_path)
        
        if results[0].boxes.xyxy is None:
            pred = image.copy()
        else:
            tensor_obj = results[0].boxes.xyxy
            numpy_obj_classes = results[0].boxes.cls.cpu().numpy()
            arr = tensor_obj.cpu().detach().numpy().astype("int")
            arr_list = list(arr)
            list_acne = []
            for i in arr_list:
                list_acne.append(tuple(i))
            img_acne_spot = face.copy()
#             img_spot = face.copy()
            list_face = list(my_tuple)
            polygon = Polygon(list_face)
            count_acne=0
            count_spot=0
            for k,cls in zip(list_acne,numpy_obj_classes):
                if cls==0 or cls==1:
                    point = Point(k[2],k[3])
                    if polygon.contains(point):
                        cv2.rectangle(img_acne_spot, (k[0],k[1]), (k[2],k[3]), (0,0,255), 2)
                        count_acne+=1
                else:
                    point = Point(k[2],k[3])
                    if polygon.contains(point):
                        cv2.rectangle(img_acne_spot, (k[0],k[1]), (k[2],k[3]), (20,20,20), 2)
                        count_spot+=1
        logger.info("OK")
        model = None
        results = None
        tensor_obj = None
        arr = None
        list_l = None
        list2 = None
    except Exception as e:
        img_acne_spot,count_acne,count_spot = False, False,False
        logger.error(f"Exception occurred in detect_acne {e}", exc_info=True)
    return img_acne_spot,count_acne,count_spot
    
def driver_acne_old(img):
    try:
        face = face_extractor(img)
        cv2.imwrite("face_extr/face_acne_old.jpg",face)
        landmarks = landmarks_roi_selection(face)
        list_landmarks = landmark_acne(face,landmarks)
        my_tuple = tuple(tuple(x) for x in list_landmarks[0])
        img_acne_spot,count_acne,count_spot = detect_acne_old("face_extr/face_acne_old.jpg",face,my_tuple)
        logger.info("OK")
    except Exception as e:
        logger.error(f"Exception occurred in calculation_acne {e}", exc_info=True)
        img_acne_spot,count_acne,count_spot= False, False, False
    return img_acne_spot,count_acne,count_spot


# Acne NEW
def detect_acne_new(img_path,face,my_tuple):
    try:
        model = YOLO(acne_model_path_new)
        results = model.predict(source=img_path, conf=0.05,iou=0.2,classes=[0,1,3,5],agnostic_nms=True,imgsz=640)
        image = cv2.imread(img_path)
        
        if results[0].boxes.xyxy is None:
            pred = image.copy()
        else:
            tensor_obj = results[0].boxes.xyxy
            numpy_obj_classes = results[0].boxes.cls.cpu().numpy()
            arr = tensor_obj.cpu().detach().numpy().astype("int")
            arr_list = list(arr)
            list_acne = []
            for i in arr_list:
                list_acne.append(tuple(i))
            img_acne_spot = face.copy()
#             img_spot = face.copy()
            list_face = list(my_tuple)
            polygon = Polygon(list_face)
            count_acne=0
            count_spot=0
            for k,cls in zip(list_acne,numpy_obj_classes):
                if cls==0 or cls==3:
                    point = Point(k[2],k[3])
                    if polygon.contains(point):
                        cv2.rectangle(img_acne_spot, (k[0],k[1]), (k[2],k[3]), (0,0,255), 2)
                        count_acne+=1
                else:
                    point = Point(k[2],k[3])
                    if polygon.contains(point):
                        cv2.rectangle(img_acne_spot, (k[0],k[1]), (k[2],k[3]), (20,20,20), 2)
                        count_spot+=1
        logger.info("OK")
        model = None
        results = None
        tensor_obj = None
        arr = None
        list_l = None
        list2 = None
    except Exception as e:
        img_acne_spot,count_acne,count_spot = False, False,False
        logger.error(f"Exception occurred in detect_acne {e}", exc_info=True)
    return img_acne_spot,count_acne,count_spot
    
def driver_acne_new(img):
    try:
        face = face_extractor(img)
        landmarks = landmarks_roi_selection(face)
        list_landmarks = landmark_acne(face,landmarks)
        my_tuple = tuple(tuple(x) for x in list_landmarks[0])
        img_acne_spot,count_acne,count_spot = detect_acne_new("face_extr/face_acne_old.jpg",face,my_tuple)
        logger.info("OK")
    except Exception as e:
        logger.error(f"Exception occurred in calculation_acne {e}", exc_info=True)
        img_acne_spot,count_acne,count_spot= False, False, False
    return img_acne_spot,count_acne,count_spot

# -----------------------------------------------------------  Age Work  ----------------------------------------------------------

def detect_age_deepface(img):
    face = face_extractor(img)
    cv2.imwrite("face_extr/face_age_deepface.jpg",face)
    image_path = "face_extr/face_age_deepface.jpg"
    try:
        objs = DeepFace.analyze(img_path = image_path, actions = ['age'],enforce_detection=False,detector_backend="mediapipe")
        age= objs[0]['age']
        a1= age-2
        a2= age+2
        result= f"{a1} - {a2}"
        logger.info("OK")
        return result, face
    except Exception as e:
        face= False
        logger.error(f"Exception occurred in calculation_acne {e}", exc_info=True)
        return "DeepFace Error", face

def detect_age_yolo_exp64(img):
    try:
        img_path = "face_extr/face_age_deepface.jpg"
        model = YOLO(age_model_path)
        results = model.predict(img_path, conf=0.01)
        list1 = list(results[0].names.values())
        ind = results[0].probs.top1
        pred = list1[ind]
        logger.info("OK")
    except Exception as e:
        logger.error(f"Exception occurred in calculation_acne {e}", exc_info=True)
        pred = False   
    return pred
#----------------------------------------------------------------------------------------------------------------------------


'''
Calculation function is for calling all the function at one Place and return response accordingly.
'''

'''
def calculation(img):
    w_old,w_new = driver_wrinkle(img)
    acne_img_old,acne_cnt_old,spot_img_old,spot_cnt_old = driver_acne_old(img)
    acne_img_new,acne_cnt_new,spot_img_new,spot_cnt_new = driver_acne_new(img)
    age_old , age_new = detect_age_deepface() , detect_age_yolo_exp64()
    
    response = {
                "old":{"wrinkle" : {"img":w_old},
                      "acne"     : {"img":acne_img_old,"score":acne_cnt_old},
                      "spot"     : {"img":spot_img_old,"score":spot_cnt_old},
                      "age"      : age_old
                      },
                "new":{"wrinkle" : {"img":w_new},
                      "acne"     : {"img":acne_img_new,"score":acne_cnt_new},
                      "spot"     : {"img":spot_img_new,"score":spot_cnt_new},
                      "age"      : age_new
                      }
               }
    return response
'''
