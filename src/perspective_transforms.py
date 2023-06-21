import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity

# from IPython.display import Image

def get_similarity_scores(input_image, side='front', nid_type='new'):
    '''
        Inputs:
            input_image = Input Image
            side = front/back
            nid_type = new/old
            
        Output:
            True => if the format matches with the template
            False => if the format doesn't match with the template
        
    '''
    
    MIN_MATCH_COUNT = 230

    '''
        ### "Caution!!!!". Check for local deployment. Maynot match with the given hostname.
    '''
    
    working_dir = os.getcwd() + '/src/' # for server (running in docker)
    
    # print('working dir in prospective transforms ifelse', working_dir)

    # 1- template, 2- image to search
    old_template_front = working_dir + "/assets/templates/old_front.png"  
    old_template_back = working_dir + "/assets/templates/old_back.png"  
    new_template_front = working_dir + "/assets/templates/new_front.png"  
    new_template_back = working_dir + "/assets/templates/new_back.png"  
    
    if nid_type == 'new':
        imgname1 = new_template_back if side =='back' else new_template_front
    else:
        imgname1 = old_template_back if side =='back' else old_template_front
        
    # imgname2 = input_image

    ## (1) prepare data
    img1 = cv2.imread(imgname1)
    # img2 = cv2.imread(imgname2)
    img2 = input_image
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


    ## (2) Create SIFT object
    sift = cv2.SIFT_create()

    ## (3) Create flann matcher
    matcher = cv2.FlannBasedMatcher(dict(algorithm = 1, trees = 5), {})

    ## (4) Detect keypoints and compute keypointer descriptors
    kpts1, descs1 = sift.detectAndCompute(gray1,None)
    kpts2, descs2 = sift.detectAndCompute(gray2,None)

    ## (5) knnMatch to get Top2
    matches = matcher.knnMatch(descs1, descs2, 2)
    # Sort by their distance.
    matches = sorted(matches, key = lambda x:x[0].distance)

    ## (6) Ratio test, to get good matches.
    good = [m1 for (m1, m2) in matches if m1.distance < 0.7 * m2.distance]

    canvas = img2.copy()

    ## (7) find homography matrix
    if len(good)>MIN_MATCH_COUNT:
#         print(len(good))
        ## (queryIndex for the small object, trainIndex for the scene )
        src_pts = np.float32([ kpts1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kpts2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        ## find homography matrix in cv2.RANSAC using good match points
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
        h,w = img1.shape[:2]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        cv2.polylines(canvas,[np.int32(dst)],True,(0,255,0),3, cv2.LINE_AA)

        ## (8) drawMatches
        matched = cv2.drawMatches(img1,kpts1,canvas,kpts2,good,None)#,**draw_params)

        ## (9) Crop the matched region from scene
        h,w = img1.shape[:2]
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv2.perspectiveTransform(pts,M)
        perspectiveM = cv2.getPerspectiveTransform(np.float32(dst),pts)
        found = cv2.warpPerspective(img2,perspectiveM,(w,h))

        # cv2.imwrite("matched.png", matched)
        # cv2.imwrite("found.png", found)

        before = img1.copy()
        after = found.copy()

        # Convert images to grayscale
        before_gray = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
        after_gray = cv2.cvtColor(after, cv2.COLOR_BGR2GRAY)

        # Compute SSIM between two images
        (score, diff) = structural_similarity(before_gray, after_gray, full=True)
        # print("Image similarity", score)
        return len(good), score, found, matched

    else:
        print( "Not enough matches are found - {}/{}".format(len(good),MIN_MATCH_COUNT))
        return 0, 0, None, None
