import utils_drtv

def template_validation_return_dict(which_template):
    return {
        "status": 200,
        "template_name": which_template,
        "message": "success",
    }


def general_return_dict(which_template, extracted_texts):
    return {
            "status": 200,
            "message": "success",
            "template_name": which_template,
            "validation_status": True,
            "extracted_texts": extracted_texts,
        }


def error_return_dict(e=None):
    return {
            "status": 400,
            "error": str(e),
            "message": "failure",
        }


def nid_return_dict(which_template, extracted_texts, dob, nid_number):
    return {
            "status": 200,
            "message": "success",
            "template_name": which_template,
            "validation_status": True,
            "extracted_texts": extracted_texts,
            "dob": dob,
            "nid_number": nid_number,
        }


def birth_certificate_return_dict(which_template, extracted_texts, birth_certificate_number, registration_date, issuance_date):
    return {
            "status": 200,
            "message": "success",
            "template_name": which_template,
            "validation_status": True,
            "extracted_texts": extracted_texts,
            "birth_certificate_number": birth_certificate_number,
            "registration_date": registration_date,
            "issuance_date": issuance_date,
        }


def driving_licence_return_dict(which_template, extracted_texts, driving_licence_number, validity_date):
    return {
            "status": 200,
            "message": "success",
            "template_name": which_template,
            "validation_status": True,
            "extracted_texts": extracted_texts,
            "driving_licence_number" : driving_licence_number,
            "validity_date": validity_date,
        }


def tin_return_dict(which_template, extracted_texts, tin_number, previous_tin, etin_status, date):
    return {
            "status": 200,
            "message": "success",
            "template_name": which_template,
            "validation_status": True,
            "extracted_texts": extracted_texts,
            "tin_number": tin_number,
            "previous_tin": previous_tin,
            "etin_status": etin_status,
            "date": date,
        }


def bin_return_dict(which_template, extracted_texts, bin_number, etin_number, issue_date, effective_date):
    return {
            "status": 200,
            "message": "success",
            "template_name": which_template,
            "validation_status": True,
            "extracted_texts": extracted_texts,
            "bin_number,": bin_number,
            "etin_number": etin_number,
            "issue_date": issue_date,
            "effective_date": effective_date,
        }


def education_return_dict(which_template, extracted_texts, education_type):
    return {
            "status": 200,
            "message": "success",
            "template_name": which_template,
            "validation_status": True,
            "extracted_texts": extracted_texts,
            "education_type": education_type,
        }


def get_specific_return_dict(image, which_template):
    if which_template == 'NID':
        extracted_texts = utils_drtv.correct_noise_and_extract_text(image)
        dob, nid_number = utils_drtv.get_dob_and_nid_number(extracted_texts)
        return nid_return_dict(which_template, extracted_texts, dob, nid_number)

    if which_template == 'BIRTH_CERTIFICATE':
        extracted_texts = utils_drtv.correct_noise_and_extract_text(image)
        birth_certificate_number, registration_date, date_of_issue = utils_drtv.get_birth_certificate_information(extracted_texts)
        return birth_certificate_return_dict(which_template, extracted_texts, birth_certificate_number, registration_date, date_of_issue)

    elif which_template in ['DL']:
        extracted_texts = utils_drtv.correct_noise_and_extract_text(image)
        driving_licence_number, validity_date = utils_drtv.get_driving_licence_information(extracted_texts)
        return driving_licence_return_dict(which_template, extracted_texts, driving_licence_number, validity_date)

    elif which_template in ['ETIN', 'E-TIN']:
        extracted_texts = utils_drtv.correct_noise_and_extract_text(image)
        tin_num, previous_tin, status, date = utils_drtv.get_tin_information(extracted_texts)
        return tin_return_dict(which_template, extracted_texts, tin_num, previous_tin, status, date)

    elif which_template in ['BIN']:
        extracted_texts = utils_drtv.correct_noise_and_extract_text(image)
        bin_num, etin_num, issue_date, effective_date = utils_drtv.get_bin_information(extracted_texts)
        return bin_return_dict(which_template, extracted_texts,bin_num, etin_num, issue_date, effective_date)

    elif which_template in ['Trade_Licence']:
        extracted_texts = utils_drtv.correct_noise_and_extract_text(image)
        return general_return_dict(which_template, extracted_texts)

    elif which_template in ['Certificate', 'Marksheet']:
        extracted_texts = utils_drtv.correct_noise_and_extract_text(image)
        education_type = utils_drtv.get_education_type(extracted_texts)
        return education_return_dict(which_template, extracted_texts, education_type)

    elif which_template in ['CS', 'RS', 'Office_Adesh', 'Shorkari_Potro', 'PASSPORT']:
        extracted_texts = utils_drtv.correct_noise_and_extract_text(image)
        return general_return_dict(which_template, extracted_texts)
    
    else:
        return {
                "status": 200,
                "template_name": which_template,
                "validation_status" : False,
                "extracted_texts": "",
                "message": "success",
            }
