import logging
from starlette.responses import HTMLResponse
from fastapi import FastAPI, Request, UploadFile, status, Response, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import io
from PIL import Image
import numpy as np
import base64
import cv2
import os
from driver import detect_age_deepface, detect_age_yolo_exp64, driver_acne_new, driver_acne_old, driver_wrinkle, face_existence, face_extractor
import mediapipe as mp

testing_app = FastAPI()


testing_app.mount(
    "/static",
    StaticFiles(directory=Path(__file__).parent.absolute() / "static"),
    name="static",
)
templates = Jinja2Templates(directory="templates")
logging.basicConfig(filename="app.log", level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")


origins = ["*"]
testing_app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
upload_dir = "uploads"
os.makedirs(upload_dir, exist_ok=True)

@testing_app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@testing_app.get("/acne_spot", response_class=HTMLResponse)
async def acne_spot(request: Request):
    return templates.TemplateResponse("acne_spot.html", {"request": request})

@testing_app.post("/acne_spot", response_class=HTMLResponse)  
async def age(request: Request, image: UploadFile):
    try:
        if image:
            content = await image.read()

            if not image.content_type.startswith("image/"):
                return templates.TemplateResponse("file_error.html", {"request": request})
                
            nparr = np.fromstring(content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            face= face_extractor(img)
            face_check= face_existence(face)
            
            if face_check is False:
                return templates.TemplateResponse("mediapipe_error.html", {"request": request})


            nparr = np.fromstring(content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # acne spot old
            result_a_o, score_a_o, score_s_o= driver_acne_old(img)
            # score_a_o= result['old']['acne_spot']['acne_score']
            result_a_o = cv2.imencode('.jpg', result_a_o)[1].tobytes()
            result_a_o = base64.b64encode(result_a_o).decode('utf-8')
            # score_s_o= result['old']['acne_spot']['spot_score']
            

            #acne spot new
            result_a_n, score_a_n, score_s_n= driver_acne_new(img)
            # score_a_n= result['new']['acne_spot']['acne_score']
            result_a_n = cv2.imencode('.jpg', result_a_n)[1].tobytes()
            result_a_n = base64.b64encode(result_a_n).decode('utf-8')
            # score_s_n= result['new']['acne_spot']['spot_score']
    except Exception as e:
        print(e)
        return templates.TemplateResponse("error.html", {"request": request})
    return templates.TemplateResponse("acne_spot.html", {"request": request, "acne_old": result_a_o, "acne_old_score": score_a_o,
                                                           "spot_old_score": score_s_o,
                                                            "acne_new": result_a_n, "acne_new_score": score_a_n,  "spot_new_score": score_s_n})

@testing_app.get("/wrinkle", response_class=HTMLResponse)  
async def wrinkle(request: Request):
        return templates.TemplateResponse("wrinkle.html", {"request": request})

@testing_app.post("/wrinkle", response_class=HTMLResponse)  
async def age(request: Request, image: UploadFile):
    try:
        if image:
            content = await image.read()

            if not image.content_type.startswith("image/"):
                return templates.TemplateResponse("file_error.html", {"request": request})
                
            nparr = np.fromstring(content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            face= face_extractor(img)
            face_check= face_existence(face)
            
            if face_check is False:
                return templates.TemplateResponse("mediapipe_error.html", {"request": request})


            nparr = np.fromstring(content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            print(img.shape)
            result_o, result_n= driver_wrinkle(img)
            
            #wrinkle old
            # result_o= result['old']['wrinkle']['img']
            result_o = cv2.imencode('.jpg', result_o)[1].tobytes()
            result_o = base64.b64encode(result_o).decode('utf-8')

            #wrinkle new 
            # result_n= result['new']['wrinkle']['img']
            result_n = cv2.imencode('.jpg', result_n)[1].tobytes()
            result_n = base64.b64encode(result_n).decode('utf-8')
    except Exception as e:
        print(e)
        return templates.TemplateResponse("error.html", {"request": request})
    return templates.TemplateResponse("wrinkle.html", {"request": request, 'wrinkle_old': result_o, "wrinkle_new": result_n})



@testing_app.get("/age", response_class=HTMLResponse)
async def age_get(request: Request):
    return templates.TemplateResponse("age.html", {"request": request})

@testing_app.post("/age", response_class=HTMLResponse)  
async def age(request: Request, image: UploadFile):
    try:
        if image:
            content = await image.read()

            if not image.content_type.startswith("image/"):
                return templates.TemplateResponse("file_error.html", {"request": request})
                
            nparr = np.fromstring(content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            face= face_extractor(img)
            face_check= face_existence(face)
            
            if face_check is False:
                return templates.TemplateResponse("mediapipe_error.html", {"request": request})


            nparr = np.fromstring(content, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            #age new
            result_o, face= detect_age_deepface(img)
            face = cv2.imencode('.jpg', face)[1].tobytes()
            face = base64.b64encode(face).decode('utf-8') 
            #age old
            result_n= detect_age_yolo_exp64(img)

            # Save the uploaded files to a temporary location
            # with open(os.path.join(upload_dir, new_model_image.filename), "wb") as new_model_file:
            #     shutil.copyfileobj(new_model_image.file, new_model_file)

            # with open(os.path.join(upload_dir, old_model_image.filename), "wb") as old_model_file:
            #     shutil.copyfileobj(old_model_image.file, old_model_file)
    except Exception as e:
        print(e)
        return templates.TemplateResponse("error.html", {"request": request})
    return templates.TemplateResponse("age.html", {"request": request, "age_old": result_o, "age_new": result_n, "face": face})
