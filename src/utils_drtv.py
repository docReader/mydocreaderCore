import os
import re
import cv2
import uuid
import requests
import numpy as np
import configparser
from time import time
from pytesseract import Output, image_to_data
from auto_noise.AutomaticNoiseRemoval import correct_noise

config = configparser.ConfigParser()
config.read(os.path.join(os.getcwd(), 'src/config.ini'))

fileserver = config.get('PATHS', 'fileserver')


def get_checkpoint(type):
    return os.path.join(os.getcwd(), config.get('CHECKPOINTS', type))


def timer_func(func):
    # This function shows the execution time of 
    # the function object passed
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2 - t1):.4f}s')
        return result

    return wrap_func


def _get_raw_image(projectID, fileName):
    image_url = fileserver + projectID + '/' + fileName + '/'
    response = requests.get(image_url, stream=True).raw
    if response.status != 200: raise Exception(
        "No such file or directory! Please try again with the correct projectID and fileName.")
    return response


@timer_func
def get_clean_image(img):
    clean_img = correct_noise(img=img, skew=1, border=1, background=1, cycleGanNR=1, uid=str(uuid.uuid1()))
    return clean_img


@timer_func
def read_image_from_fileserver(projectID, fileName, noiseRemove=False):
    print("\nReading from ProjectID - {} ; FileName - {}".format(projectID, fileName))
    response = _get_raw_image(projectID, fileName)
    image = np.asarray(bytearray(response.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if noiseRemove:
        return get_clean_image(image)
    else:
        return image


@timer_func
def extract_text(img):
    """
      Extract text leveraging tesseract.

      Parameters:
        gray_image (image array):this part will take gray image as input

      Returns:
        image array:Returning English removed image
    """
    # img = cv2.imread(img)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 21, 15)
    # if nidty == "new":
    #   img[img > 90] = 255 # change everything to white where pixel is not black #66
    # else:
    #   img[img > 156] = 255 # change everything to white where pixel is not black #66

    custom_config = r'-l eng+ben --psm 11'
    results = image_to_data(image=img, output_type='data.frame',
                            config=custom_config)  # previous output_type=Output.dict
    # texts = list(dict.fromkeys(results["text"]))

    results = results[results.conf != -1]  # remove blanks
    lines = results.groupby(['page_num', 'block_num', 'par_num', 'line_num'])['text'].apply(
        lambda x: ' '.join(map(str, x))).tolist()

    texts = ""
    for line in lines:
        texts = texts + line.strip() + "\n"

    return texts


def correct_noise_and_extract_text(img):
    return extract_text(get_clean_image(img))


@timer_func
def get_dob_and_nid_number(extracted_texts):
    dob_pattern = '((0?[1-9])|1\d|2[0-9]) (Jan|Feb|Mar|May|Apr|Jul|Jun|Aug|Oct|Sep|Nov|Dec) ((1[6-9]|[2-9]\d)\d{2})'
    new_nid_pattern = '(\d{3} \d{3} \d{4})'
    old_nid_pattern1 = '(\d{17})'  # for 17 digit nid
    old_nid_pattern2 = '(\d{13})'  # for 13 digit nid
    temporary_nid_pattern = '(\d{10})'  # for 10 digit temporary nid

    result_dob = re.search(dob_pattern, extracted_texts)
    result_nid = re.search(new_nid_pattern, extracted_texts)
    if result_nid == None: result_nid = re.search(old_nid_pattern1, extracted_texts)
    if result_nid == None: result_nid = re.search(old_nid_pattern2, extracted_texts)
    if result_nid == None: result_nid = re.search(temporary_nid_pattern, extracted_texts)

    result_dob = result_dob.group() if result_dob != None else ''
    result_nid = result_nid.group().replace(' ', '') if result_nid != None else ''

    return result_dob, result_nid


@timer_func
def get_driving_licence_information(extracted_texts):
    driving_licence_pattern = '[A-Z]{2}[A-Z0-9]{13}'
    validity_date_pattern = '\d{2}\s[A-Z]{3}\s\d{4}'

    driving_licence_number = re.search(driving_licence_pattern, extracted_texts)
    validity_date = re.findall(validity_date_pattern, extracted_texts)
    driving_licence_number = driving_licence_number.group() if driving_licence_number != None else ''
    validity_date = validity_date[-1]

    return driving_licence_number, validity_date



def _find_birth_certificate_number(extracted_texts):
    splits = extracted_texts.split('\n')
    target_line = ""
    birth_certificate_number = ""

    for split in splits:
        numbers = sum(c.isdigit() for c in split)
        if numbers > 12:
            target_line = split

    print(target_line)

    for c in target_line:
        if c.isdigit():
            birth_certificate_number = birth_certificate_number + c
            
    return birth_certificate_number.strip()


def _find_registration_date_and_date_of_issue_of_birth_certificate(extracted_texts):
    registration_date_pattern = "(0?[1-9]|[12][0-9]|3[01])[\/\-](0?[1-9]|1[012])[\/\-]\d{4}"
    issuance_date_pattern = "\d{2}-\d{2}-\d{4}"
    registration_date_pattern_online = '\d{2}\s[A-Z]{3,12}\s\d{4}'
    issuance_date_pattern_online = '\d{2}\s[A-Z]{3,12}\s\d{4}'
    
    #finding registration date
    result_reg_date = re.search(registration_date_pattern, extracted_texts)
    if result_reg_date == None: result_reg_date = result_reg_date = \
                        re.search(registration_date_pattern_online, extracted_texts)
    registration_date = result_reg_date.group() if result_reg_date != None else ''
    
    # finding date of issue
    result_issue_date = re.findall(issuance_date_pattern, extracted_texts)
    if len(result_issue_date) >= 1:
        issuance_date = result_issue_date[1] if len(result_issue_date) >= 1 else ''

    else: # finding for online copy
        result_issue_date = re.findall(issuance_date_pattern_online, extracted_texts)
        issuance_date = result_issue_date[1] if len(result_issue_date) >= 1 else ''

    return registration_date.strip(), issuance_date.strip()


@timer_func
def get_birth_certificate_information(extracted_texts):
    birth_certificate_number = _find_birth_certificate_number(extracted_texts)
    registration_date, issuance_date = \
            _find_registration_date_and_date_of_issue_of_birth_certificate(extracted_texts)
    return birth_certificate_number, registration_date, issuance_date


# @timer_func
#def get_tin_number(extracted_texts):
#    tin_number_pattern = '(\d{12})'  # for 12 digit etin certificate.
#    result_tin = re.search(tin_number_pattern, extracted_texts)
#    tin_number = result_tin.group() if result_tin != None else ''

    #return tin_number
    

@timer_func
def get_tin_information(extracted_texts):
    tin_pattern = '(\d{12})'
    previous_tin_pattern = '(?<=Previous TIN :)(.(\w+).(\w+)(?:\s|.))'
    status_patter = '(?<=Status :)(.(\w+)(?:\s|.))'
    date_pattern =  '\w+\s\d{2},\s\d{4}'

    result_tin_num = re.search(tin_pattern, extracted_texts)
    result_previous_tin = re.search(previous_tin_pattern, extracted_texts)
    result_status = re.search(status_patter, extracted_texts)
    result_date = re.search(date_pattern, extracted_texts)

    tin_number = result_tin_num.group().strip() if result_tin_num != None else ''
    previous_tin = result_previous_tin.group().strip() if result_previous_tin != None else ''
    etin_status = result_status.group().strip() if result_status != None else ''
    date = result_date.group().strip() if result_date != None else ''

    previous_tin = previous_tin.replace('\n', '') if '\n' in previous_tin else previous_tin
    etin_status = etin_status.replace('\n', '') if '\n' in etin_status else etin_status

    return tin_number, previous_tin, etin_status, date


@timer_func
def get_bin_information(extracted_texts):
    extracted_texts = extracted_texts.replace("\n"," ")
    bin_number_pattern = '\d{9}-\d{4}'
    etin_number_pattern = '\d{12}'
    issue_date_pattern = "(?<=Issue Date).*?([0-9]|1[0-9]|2[0-9]|3[0-5])/([0-9]|1[0-9]|2[0-9]|3[0-5])/([0-9][0-9][0-9][0-9])"
    effective_date_pattern = "(?<=Effective Date).*?([0-9]|1[0-9]|2[0-9]|3[0-5])/([0-9]|1[0-9]|2[0-9]|3[0-5])/([0-9][0-9][0-9][0-9])"

    result_bin_num = re.search(bin_number_pattern, extracted_texts)
    result_etin_num = re.search(etin_number_pattern, extracted_texts)
    result_issue_date = re.search(issue_date_pattern, extracted_texts)
    result_effective_date = re.search(effective_date_pattern, extracted_texts)

    bin_number = result_bin_num.group().strip() if result_bin_num != None else ''
    etin_number = result_etin_num.group().strip() if result_etin_num != None else ''
    issue_date = result_issue_date.group().strip() if result_issue_date != None else ''
    effective_date = result_effective_date.group().strip() if result_effective_date != None else ''

    return bin_number, etin_number, issue_date, effective_date


@timer_func
def get_education_type(extracted_texts):
    type = ''
    if 'secondary school certificate' in extracted_texts.lower():
        type = 'SSC'
    if 'higher secondary' in extracted_texts.lower():
        type = 'HSC'

    return type


def check_keywords_from_extracted_text(txt, nid_side):
    for i in range(len(txt)):
        nid_flag = False

        if nid_side == 'back':
            if "ডাকঘর" in txt[i]:
                nid_flag = True
                break

        else:
            if ("জাতীয়" in txt[i] or "জাতায়" in txt[i]) and "পরিচয়পত্র" in txt[i + 1]:
                nid_flag = True
                break

            elif ("জাতীয়" in txt[i] or "জাতায়" in txt[i]) and txt[i + 1] == "পরিচয়" and txt[i + 2] == "পত্র":
                nid_flag = True
                break
            elif txt[i] == "National" and txt[i + 1] == "ID" and txt[i + 2] == "Card":
                nid_flag = True
                break

    return nid_flag
