import cv2
import numpy as np
import os
from auto_noise.CycleGanNoiseRemoval import correct_noise_cycleGan
from tensorflow.keras.preprocessing import image
from datetime import datetime

"""
####################################################################################################
                                         SKEW CORRECTION
####################################################################################################
"""
# import resource
# def using(point=""):
#     usage=resource.getrusage(resource.RUSAGE_SELF)
#     return '''%s: usertime=%s systime=%s mem=%s mb
#            '''%(point,usage[0],usage[1],
#                 usage[2]/1024.0 )

"""
Image Resize for cycleGAN
"""
def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):

    dim = None
    (h, w) = image.shape[:2]
    
    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
        
    else:
        
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation = inter)

    return resized





def process_image(img):
    """
    Some image processing is done inside (Gaussian
    Blur, OTSU Binarization and Dilation)

    Parameters
    ----------
    img as CV/Numpy Image

    Returns
    -------
    img as CV/Numpy Image
    """

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = cv2.GaussianBlur(img, (7, 7), 0)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 1))
    img = cv2.dilate(img, kernel, iterations=5)

    # cv2.imshow('window', img)
    # cv2.waitKey()

    return img


def rotate_image(img, angle: float):
    """
    Rotates an Image for a given angle

    Parameters
    ----------
    img as CV/Numpy Image
    angle as a float

    Returns
    -------
    img as CV/Numpy Image
    """

    h, w = img.shape[:2]

    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    img = cv2.warpAffine(
        img, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255)
    )

    return img


def find_angle(img):
    """
    Finds the skewness present in an image in terms of angle

    Parameters
    ----------
    img as CV/Numpy Image

    Returns
    -------
    result_angle as float

    """

    # tmp_img = img.copy()
    img = process_image(img)

    contours = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]

    considered_contours = []
    contour_properties = []
    widths = []
    heights = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        angle = cv2.minAreaRect(contour)[-1]
        contour_properties.append([x, y, w, h, angle])
        considered_contours.append(contour)
        widths.append(w)
        heights.append(h)
        contour_area = w * h
        # if (w > 3 * h) and (contour_area >= 0.01 * img_area):

        # if abs(angle) >= 45:
        #     ctrs_for_print.append(contour)

        # angles.append(angle)

    # print(considered_contours)
    # print(widths)
    # print("--------------------------------------------------")
    # print(heights)
    mean_width = np.mean(widths)
    # print("Mean Width: ", mean_width)
    std_width = np.std(widths)

    mean_height = np.mean(heights)
    # print("Mean Height: ", mean_height)
    std_height = np.std(heights)
    # print("Mean Height: ", mean_height)


    # print("#########################################################")
    # print("Len of widths: ", len(widths))
    # print("Len of contour properties: ", len(contour_properties))
    # print("Len of considered Contours: ", len(considered_contours))
    # print("#########################################################")

    considered_angles = []
    tmp_ctr = []

    for i, width in enumerate(widths):
        # tmp_img = cv2.drawContours(tmp_img, considered_contours[i], 0, (0, 255, 0), 5)
        # if width >= (mean_width + std_width):
        if (width >= mean_width) and (heights[i] <= (mean_height + 2.5*std_height)):
            # print("HERE")

            # tmp_angle = contour_properties[i][-1]
            #
            # if tmp_angle < -45:
            #     tmp_angle = 90 + tmp_angle

            tmp_angle = contour_properties[i][-1]
            #print(tmp_angle)

            if tmp_angle < -45:
                tmp_angle = 90 + tmp_angle

            elif tmp_angle > 45:
                tmp_angle = tmp_angle - 90

            if abs(tmp_angle) < 45:
                considered_angles.append(tmp_angle)
                tmp_ctr.append(considered_contours[i])

    if(len(considered_angles)==0):
        return 0
    #print("Considered Angles: ", considered_angles)

    # print("Length Considered Angle: ", len(considered_angles))
    # print("Length Tmp Contour: ", len(tmp_ctr))

    # tmp_img = cv2.drawContours(tmp_img, tmp_ctr, -1, (0, 255, 0), 3)
    # print("Count of Considered Contours: ", len(considered_contours))
    # cv2.imshow('window', tmp_img)
    # cv2.waitKey()

    # if len(considered_angles) < 1:
    #     considered_angles.append([0])

    # print("Considered Angles: ", considered_angles)

    data_mean, data_std = np.mean(considered_angles), np.std(considered_angles)
    cut_off = data_std * 1
    lower, upper = data_mean - cut_off, data_mean + cut_off

    skew_angles = []

    # tmp2_ctr = []

    for i, angle in enumerate(considered_angles):
        if abs(angle) < 44:
            if (lower) <= angle <= (upper):
                skew_angles.append(angle)
                # tmp2_ctr.append(tmp_ctr[i])

        # if abs(angle) > 45:
        #     tmp2_ctr.append(tmp_ctr[i])

    # for tmp2 in tmp2_ctr:
    #     rect = cv2.minAreaRect(tmp2)
    #     box = cv2.boxPoints(rect)
    #     box = np.int0(box)
    #     tmp_img = cv2.drawContours(tmp_img, [box], 0, (0, 255, 0), 3)

    # tmp_img = cv2.drawContours(tmp_img, tmp2_ctr, -1, (0, 255, 0), 3)
    #print("Skew Angles: ", skew_angles)
    result_angle = np.mean(skew_angles)


    # result_angle = np.mean(considered_angles)

    if result_angle < -45:
        result_angle = 90 + result_angle

    return result_angle


def correct_skew(img):
    """
    Corrects the skewness present in an image

    Parameters
    ----------
    img as CV/Numpy Image

    Returns
    -------
    img as CV/Numpy Image
    """

    angle = find_angle(img)
    final_img = rotate_image(img, angle)

    # print("Angle: ", angle)

    return final_img


"""
####################################################################################################
                                    BORDER NOISE CORRECTION
####################################################################################################
"""
def textAreaSelection(img):
    #img_path = os.path.join(input_loc, file)
    canny_img =img
    #img =canny_img = cv2.imread(img_path, 1)
    #img = canny_img
    canny_img = process_image(canny_img)
    contours = cv2.findContours(canny_img,cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)[0]

    considered_contours = []
    contour_properties = []
    widths = []
    heights = []
    xs = []
    ys = []
    contour_area=[]
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        angle = cv2.minAreaRect(contour)[-1]
        contour_properties.append([x, y, w, h, angle])
        considered_contours.append(contour)
        xs.append(x)
        ys.append(y)
        widths.append(w)
        heights.append(h)
        contour_area.append(w*h)
        #cv2.rectangle(img, (x,y), (x+w,y+h), (255, 0, 0), 2)


    mean_width = np.mean(widths)
    std_width = np.std(widths)
    mean_height = np.mean(heights)
    std_height = np.std(heights)
    considered_x = []
    xs = []
    ys = []
    max_x = 0
    max_y = 0
    widths_up=[]
    heights_up=[]
    for i, width in enumerate(widths):
        if (width >= mean_width-std_width*3) and ((mean_height - std_height*3) <= heights[i] <= (mean_height + 3*std_height)):

            widths_up.append(width)
            x,y,w,h=contour_properties[i][0],contour_properties[i][1],contour_properties[i][2],contour_properties[i][3]
            considered_x.append([x,y,w,h])
            xs.append(x)
            ys.append(y)
            heights_up.append(h)

            if max_x<x+w:
                max_x=x+w
            if max_y<y+h:
                max_y=y+h

    widths=widths_up
    heights=heights_up
    mean_width = np.mean(widths)
    std_width = np.std(widths)
    mean_height = np.mean(heights)
    std_height = np.std(heights)
    xs = []
    ys = []
    max_x = 0
    max_y = 0

    #print("considered_x")
    for i, width in enumerate(widths):
        if (width >= mean_width-std_width*0.6) and ((mean_height - std_height*0.3) <= heights[i] <= (mean_height + 3*std_height)):
            x,y,w,h=considered_x[i][0],considered_x[i][1],considered_x[i][2],considered_x[i][3]
            xs.append(x)
            ys.append(y)
            if max_x<x+w:
                max_x=x+w
            if max_y<y+h:
                max_y=y+h

    acl_y,acl_x,_ = img.shape

    min_x = np.min(xs)
    min_x = int(min_x*.60)
    max_x = int(max_x+(acl_x-max_x)*.5)

    min_y = np.min(ys)
    min_y = int(min_y*.60)
    max_y = int(max_y+(acl_y-max_y)*.5)


    if(max_y>acl_y*0.99):
        max_y = int(acl_y*0.99)
    if(max_x>acl_x*0.99):
        max_x = int(acl_x*0.99)
    if(min_y<acl_y*0.01):
        min_y = int(acl_y*0.01)
    if(min_x<acl_x*0.01):
        min_x = int(acl_x*0.01)

    img = img[min_y:max_y,min_x:max_x]
    
    return img

"""
def remove_border_noise(img, borderY=30, borderX=60):

    img_shape = img.shape
    x, y = img_shape[1], img_shape[0]
    crop_img = img[borderY:y - borderY, borderX:x - borderX]
    # crop_img = [x + [0,0,0] for x in crop_img
    for i in range(0, borderX):
        for j in range(0, y):
            img[j][i] = [255, 255, 255]

    for i in range(x - borderX, x):
        for j in range(0, y):
            img[j][i] = [255, 255, 255]

    for i in range(0, x):
        for j in range(0, borderY):
            img[j][i] = [255, 255, 255]

    for i in range(0, x):
        for j in range(y - borderY, y):
            img[j][i] = [255, 255, 255]

    return img
    """


"""
####################################################################################################
                                    BACKGROUND NOISE CORRECTION
####################################################################################################
"""


def clean_bg(img):
    """
    Takes input location as input.
    Increases contrast of each image by 2-fold
    and subtracts brightness by 160.
    Then clips the pixel values within 0 --> 255.
    Stores the output images in the output location
    """

    alpha = 2.0
    beta = -160

    for i in range(2):
        new_img = alpha * img + beta
        new_img = np.clip(new_img, 0, 255).astype(np.uint8)
        img = new_img

    return img


def correct_noise(
        img=None,
        skew=0,
        border=0,
        background=0,
        cycleGanNR=0,
        uid = ''
):
    """
    API for accessing the pre-processing functionalities
    Parameters
    ----------
    skew, pass skew=1, if you want skew correction.
    border, call with border=1, if you border noise correction.
    background, pass background=2, if you want background noise correction.
    img, the cv/numpy image we want to pre-process.

    Returns
    -------
    Returns the pre-processed image as output.
    """
    try:
        # print('parameters', skew, border, background, cycleGanNR, uid)
        if border:
            # border_start = datetime.now()
            img = textAreaSelection(img)
            # print('border')

        if(img.shape[0]>2400):
            img = image_resize(img, height = 2400)
        if(img.shape[1]>2400):
            #print(file)
            w=int(img.shape[1]*0.75)
            img = image_resize(img, width = w)

        if skew:
            # img = image.img_to_array(img, dtype='uint8')
            # skew_start = datetime.now()
            img = correct_skew(img)
            # print('skew')


        if cycleGanNR:
            # cycleGAN_start = datetime.now()
            img = correct_noise_cycleGan(img)
            
            # if not os.path.exists('./src/auto_noise/NoiseRemovedImages'):
            #         os.makedirs('./src/auto_noise/NoiseRemovedImages')
            # cv2.imwrite('./src/auto_noise/NoiseRemovedImages/'+uid+"_cycleGAN_NR.png", img)
            # img = cv2.imread('./src/auto_noise/NoiseRemovedImages/'+uid+"_cycleGAN_NR.png")
            if img.ndim != 3:
                img=cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            # os.remove('./src/auto_noise/NoiseRemovedImages/'+uid+"_cycleGAN_NR.png")
            # print('cycleGanNR')

        # if skew:
        #     img = correct_skew(img)

        if background:
            # bg_start = datetime.now()
            img = clean_bg(img)
            # print('background')
        
        
        return img
    except Exception as e:
        print(e)
        return img

