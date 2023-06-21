import os
import cv2
from datetime import datetime
from numpy.core.numeric import Inf
from utils_drtv import timer_func
from perspective_transforms import get_similarity_scores

@timer_func
def run(image):
    # print('run.py er cwd = ', os.getcwd())
    start_time = datetime.now()
    sides = ['front', 'back']
    nid_types = ['old', 'new']

    nid_templates = ['NID (Old Front)', 'NID (Old Back)', 'NID (Smart Card Front)', 'NID (Smart Card Back)']
    matching_scores = []   # matching_scores = [old front, old back, new front, new back]
    semantic_similarity_scores = []   # semantic_similarity_scores = [old front, old back, new front, new back]
    found_images = []
    match_visualizations = []

    # get_similarity_scores and images for all four possible combinations
    for nid_type in nid_types:
        for side in sides:
            match_count, similarity_score, found, matched = get_similarity_scores(input_image=image, side=side, nid_type=nid_type)
            matching_scores.append((match_count, similarity_score))
            semantic_similarity_scores.append(similarity_score)
            found_images.append(found)
            match_visualizations.append(matched)

    print(matching_scores)

    # compare structural similarities and guess the initial template without tesseract
    try:
        ss_maxval = max(semantic_similarity_scores) if max(semantic_similarity_scores) > 0.3 else Inf
        probable_template_index = semantic_similarity_scores.index(ss_maxval)
        which_template = nid_templates[probable_template_index]

        # save images
        cv2.imwrite("matched.png", match_visualizations[probable_template_index])
        cv2.imwrite("found.png", found_images[probable_template_index])

        # get the predicted nid side (# 0,2 = front; 1,3 = back)
        if probable_template_index == 0 or probable_template_index == 2:
            which_side = sides[0]
        elif probable_template_index == 1 or probable_template_index == 3:
            which_side = sides[1]
        else: 
            which_side = 'not_nid'

    except: which_template = 'Sorry! No template matched for the given document.'
    print(which_template)


    '''
    # third step verification: Match Keywords using Tesseract
    if which_template == 'Sorry! No template matched for the given document.':
        print('Please provide a valid image.')

    elif which_side == 'back':
        print('Backside so ignoring tesseract.')

    else:
        print('Starting Tesseract! Hold on!!!')
        start_tesseract=datetime.now()
        nid_text = extract_text('./found.png')
        print('Tesseract Runtime:', datetime.now()-start_tesseract)
        
        is_nid = check_keywords_from_extracted_text(nid_text, which_side)
        nid_naki = 'NID' if is_nid else 'Not NID'

        print()
        print(nid_naki)
    '''

    print('Total Runtime:', datetime.now()-start_time)

    # return matching_scores, which_template, nid_text, nid_naki
    return None, which_template, None, 'NID' if which_template != 'Sorry! No template matched for the given document.' else "Not NID"


if __name__ == "__main__":
    nid_img_path = ''
    run(nid_img_path)


# provide the path of the input image in the "im" variable.   
# provide the side information in the side variable.
# provide nid type in the nid_type variable.


# img_path = "./data/fakru.png" 
# side = 'front'
# nid_type = 'old'

# match_count, similarity_score = get_similarity_scores(input_image=img_path, side=side, nid_type=nid_type)

# print()
# print("Match Score", match_count)
# print('Semantic Similarity Score:', similarity_score)


# if similarity_score > 0.5:
#     print('Going to Tesseracrt. Brace Yourself!')
    
#     start=datetime.now()
#     nid_text = extract_text('./found.png')
#     print('Tesseract Runtime:', datetime.now()-start)
    
#     is_nid = check_keywords_from_extracted_text(nid_text)
#     nid_naki = 'NID' if is_nid else 'Not NID'

#     print('\n\n')
#     print()
#     print(nid_naki)
    
# else:
#     print('Not a valid image.')