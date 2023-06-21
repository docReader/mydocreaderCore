import io
import os
import cv2
import torch
import shutil
import uvicorn
import configparser
from starlette.responses import StreamingResponse
        

import utils_drtv
from run import run
import infer
from infer import detect_template
import return_dicts

from fastapi import FastAPI, Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware

config = configparser.ConfigParser()
config.read(os.path.join(os.getcwd(), 'src/config.ini'))


# The flowing code will register your server to eureka server and also start to send heartbeat every 30 seconds
# import py_eureka_client.eureka_client as eureka_client
# dr_core_rest_server_port = 9852
# eureka_client.init(eureka_server="http://172.17.133.15:8761/eureka",
#                    app_name="dr-core",
#                    instance_port=dr_core_rest_server_port)

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return "Welcome to Document Reader and Template Validator."


# auto noise api
@app.post("/auto_noise/")
async def auto_noise(projectID: str = Form(...), fileName: str = Form(...)):
    try:
        image = utils_drtv.read_image_from_fileserver(projectID, fileName)
        image = utils_drtv.get_clean_image(image)
        res, im_png = cv2.imencode(".png", image)
        
        return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")
      
    except Exception as e:
        return return_dicts.error_return_dict(e)


# for template verification
@app.post("/template_validation/") 
async def template_validation(projectID: str = Form(...), fileName: str = Form(...)):
    try:
        image = utils_drtv.read_image_from_fileserver(projectID, fileName)
        # matching_scores, which_template, nid_text, nid_naki = run(image) # template_matching
        which_template, template_confidence = detect_template(image, model) # yolov5

        return return_dicts.template_validation_return_dict(which_template)

    except Exception as e:
        return return_dicts.error_return_dict(e)

# for extracting text from image
@app.post("/extract_text/")
async def extract_text(projectID: str = Form(...), fileName: str = Form(...)):
    try:
        image = utils_drtv.read_image_from_fileserver(projectID, fileName, noiseRemove=True)
        extracted_texts = utils_drtv.extract_text(image)

        return {
            "status": 200,
            "extracted_texts": extracted_texts,
            "message": "success",
        }
    except Exception as e:
        return return_dicts.error_return_dict(e)


# template validation and text extraction in single api
@app.post("/drtv_server/")
async def drtv_server(projectID: str = Form(...), fileName: str = Form(...)):
    try:
        image = utils_drtv.read_image_from_fileserver(projectID, fileName)
        # matching_scores, which_template, nid_text, nid_naki = run(image) # template matching
        which_template, template_confidence = detect_template(image, model) # yolov5

        # return get_specific_return_dict(projectID, fileName, which_template)
        return return_dicts.get_specific_return_dict(image, which_template)
        
    except Exception as e:
        return return_dicts.error_return_dict(e)


# education template validation and text extraction in single api
@app.post("/education/")
async def education(projectID: str = Form(...), fileName: str = Form(...)):
    try:
        image = utils_drtv.read_image_from_fileserver(projectID, fileName)
        which_template = infer.detect_education_template(education_model, image) # resnet18

        return return_dicts.get_specific_return_dict(image, which_template)
      
    except Exception as e:
        return return_dicts.error_return_dict(e)

# for education template verification
@app.post("/education_template_validation/") 
async def education_template_validation(projectID: str = Form(...), fileName: str = Form(...)):
    try:
        image = utils_drtv.read_image_from_fileserver(projectID, fileName)
        which_template = infer.detect_education_template(education_model, image) # resnet18

        return return_dicts.template_validation_return_dict(which_template)

    except Exception as e:
        return return_dicts.error_return_dict(e)


# business template validation and text extraction in single api
@app.post("/business/")
async def business(projectID: str = Form(...), fileName: str = Form(...)):
    try:
        image = utils_drtv.read_image_from_fileserver(projectID, fileName)
        which_template = infer.detect_business_template(business_model, image) # resnet18

        return return_dicts.get_specific_return_dict(image, which_template)
      
    except Exception as e:
        return return_dicts.error_return_dict(e)


# for business template verification
@app.post("/business_template_validation/") 
async def business_template_validation(projectID: str = Form(...), fileName: str = Form(...)):
    try:
        image = utils_drtv.read_image_from_fileserver(projectID, fileName)
        which_template = infer.detect_business_template(business_model, image) # resnet18

        return return_dicts.template_validation_return_dict(which_template)

    except Exception as e:
        return return_dicts.error_return_dict(e)



# land template validation and text extraction in single api
@app.post("/land/")
async def land(projectID: str = Form(...), fileName: str = Form(...)):
    try:
        image = utils_drtv.read_image_from_fileserver(projectID, fileName)
        which_template = infer.detect_land_template(land_model, image) # resnet18

        return return_dicts.get_specific_return_dict(image, which_template)
      
    except Exception as e:
        return return_dicts.error_return_dict(e)


# for land template verification
@app.post("/land_template_validation/") 
async def land_template_validation(projectID: str = Form(...), fileName: str = Form(...)):
    try:
        image = utils_drtv.read_image_from_fileserver(projectID, fileName)
        which_template = infer.detect_land_template(land_model, image) # resnet18

        return return_dicts.template_validation_return_dict(which_template)

    except Exception as e:
        return return_dicts.error_return_dict(e)



# nothi template validation and text extraction in single api
@app.post("/nothi/")
async def nothi(projectID: str = Form(...), fileName: str = Form(...)):
    try:
        image = utils_drtv.read_image_from_fileserver(projectID, fileName)
        which_template = infer.detect_nothi_template(nothi_model, image) # resnet18

        return return_dicts.get_specific_return_dict(image, which_template)
      
    except Exception as e:
        return return_dicts.error_return_dict(e)


# for nothi template verification
@app.post("/nothi_template_validation/") 
async def nothi_template_validation(projectID: str = Form(...), fileName: str = Form(...)):
    try:
        image = utils_drtv.read_image_from_fileserver(projectID, fileName)
        which_template = infer.detect_nothi_template(nothi_model, image) # resnet18

        return return_dicts.template_validation_return_dict(which_template)

    except Exception as e:
        return return_dicts.error_return_dict(e)


#################################### Plugin APIs #######################################################

# Template validation plugin
@app.post("/template_validation_plugin/") 
async def image(image: UploadFile = File(...)):
    try:
        with open("uploaded_file", "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        image = cv2.imread('uploaded_file')
        os.remove('uploaded_file')
        print(image.shape)

        which_template, template_confidence = detect_template(image, model) # yolov5

        return return_dicts.template_validation_return_dict(which_template)
    except Exception as e:
        return return_dicts.error_return_dict(e)


# template validation and text extraction in single api PLUGIN version
@app.post("/drtv_server_plugin/")
async def image(image: UploadFile = File(...)):
    try:
        with open("uploaded_file", "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        image = cv2.imread('uploaded_file')
        os.remove('uploaded_file')
        print(image.shape)

        which_template, template_confidence = detect_template(image, model) # yolov5

        # return get_specific_return_dict(projectID, fileName, which_template)
        return return_dicts.get_specific_return_dict(image, which_template)
    except Exception as e:
        return return_dicts.error_return_dict(e)


# Extract Text plugin
@app.post("/ocr_plugin/") 
async def image(image: UploadFile = File(...)):
    try:
        with open("uploaded_file", "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        image = cv2.imread('uploaded_file')
        os.remove('uploaded_file')
        print(image.shape)

        extracted_texts = utils_drtv.correct_noise_and_extract_text(image)

        return {"extracted_texts": extracted_texts}
    except Exception as e:
        return return_dicts.error_return_dict(e)


@app.get("/test/")
def test():
    return "Welcome to the test api"

@app.post("/test/")
async def login(username: str = Form(...), password: str = Form(...)):
    name = 'My name is ' + username.upper() 
    passcode = 'My password is ' + '*'*len(password)
    return {
        "username": name,
        "password": passcode
    } 


if __name__ == "__main__":
    model = torch.hub.load('ultralytics/yolov5', 'custom', path = utils_drtv.get_checkpoint('identity'))
    education_model = infer.load_education_model(path = utils_drtv.get_checkpoint('education'))
    business_model = infer.load_business_model(path = utils_drtv.get_checkpoint('business'))
    land_model = infer.load_land_model(path = utils_drtv.get_checkpoint('land'))
    nothi_model = infer.load_nothi_model(path = utils_drtv.get_checkpoint('nothi'))
    uvicorn.run(app, host="0.0.0.0", port=9852)