import os
import cv2
import torch
import torchvision
import configparser
from torchvision import transforms
from utils_drtv import timer_func

config = configparser.ConfigParser()
config.read(os.path.join(os.getcwd(), 'src/config.ini'))


def get_loaded_model():
    # Model
    print(os.path.join(os.getcwd(), 'assets/best.pt'))
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join(os.getcwd(), 'assets/best.pt'))  # or yolov5m, yolov5l, yolov5x, custom
    return model

@timer_func
def detect_template(img, model):
    # model = torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join(os.getcwd(), 'assets/best.pt'))  # or yolov5m, yolov5l, yolov5x, custom
    # Inference
    results = model(img)

    # Results
    # results.print()  # or .show(), .save(), .crop(), .pandas(), etc.

    # results.show()

    df = results.pandas()
    # print(df.xyxy[0])

    try:
        template_name = df.xyxy[0].iloc[0]['name'].upper()
        template_confidence = df.xyxy[0].iloc[0]['confidence']
        if template_name=='ETIN' or template_confidence < 0.7:
            template_name = 'Sorry! No template matched for the given document.'
    except:
        template_name = 'Sorry! No template matched for the given document.'
        template_confidence = 0

    return template_name, template_confidence



def load_resnet18_model(path, n_class):
    model = torchvision.models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(in_features=512, out_features=int(n_class), bias=True)
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model = model.eval()
    return model


def preprocess_for_inference(img, entity):
    inp_h = int(config.get(entity, 'inp_h'))
    inp_w = int(config.get(entity, 'inp_w'))

    # img = cv2.imread(img)
    img_h, img_w, img_c = img.shape
    img = cv2.resize(img, (0,0), fx=inp_w / img_w, fy=inp_h / img_h, interpolation=cv2.INTER_CUBIC)   
    transform = transforms.ToTensor()
    img = transform(img).unsqueeze(0)
    
    return img


def detect_template_resnet18(model, img, entity):
    img = preprocess_for_inference(img, entity)
    model_predictions = model(img)

    classes = list(config.get(entity, 'classes').split(","))
    
    _, predicted = torch.max(model_predictions, 1)

    prediction = predicted.cpu().numpy()
    prediction = classes[int(prediction)].strip()
    
    return prediction 


## ----------------------------------------------- EDUCATION (Certificate/Marksheet/Others)------------------------------------------------
def load_education_model(path):
    return load_resnet18_model(path, n_class=config.get('EDUCATION', 'n_class'))

@timer_func
def detect_education_template(model, img):
    return detect_template_resnet18(model, img, 'EDUCATION')
## ----------------------------------------------------------- EDUCATION ENDS------------------------------------------------

## ----------------------------------------------- BUSINESS (BIN/TIN/TRADE_LICENCE)------------------------------------------------
def load_business_model(path):
    return load_resnet18_model(path, n_class=config.get('BUSINESS', 'n_class'))

@timer_func
def detect_business_template(model, img):
    return detect_template_resnet18(model, img, 'BUSINESS')
## ----------------------------------------------------------- BUSINESS ENDS------------------------------------------------

## ----------------------------------------------- LAND (CS/RS/Others)------------------------------------------------
def load_land_model(path):
    return load_resnet18_model(path, n_class=config.get('LAND', 'n_class'))

@timer_func
def detect_land_template(model, img):
    return detect_template_resnet18(model, img, 'LAND')
## ---------------------------------------------------- LAND ENDS----------------------------------------------------


## ----------------------------------------------- NOTHI (Office_Adesh/ Shorkati_Potro/ Others)------------------------------------------------
def load_nothi_model(path):
    return load_resnet18_model(path, n_class=config.get('NOTHI', 'n_class'))

@timer_func
def detect_nothi_template(model, img):
    return detect_template_resnet18(model, img, 'NOTHI')
## ---------------------------------------------------- NOTHI ENDS----------------------------------------------------


if __name__ == '__main__':
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=os.path.join(os.getcwd(), 'assets/ckpts/yolov5/nid_dl_etin.pt'))  # or yolov5m, yolov5l, yolov5x, custom
    template_name, template_confidence = detect_template('/home/sazzad/Documents/GitHub/a2i-Document-Template-Verification/src/found.png', model)
    print(template_name, template_confidence)