import numpy as np
import cv2 as cv

def key_features_in_image(image):

    # image to grayscale and numpy
    image_gray = cv.cvtColor(src=image, code=cv.COLOR_RGB2GRAY)
    image_gray = np.array(image_gray)

    # detect feature in the image and calculate their descriptor
    sift = cv.SIFT_create()
    kp, des = sift.detectAndCompute(image_gray, None)

    return kp, des

def match_features_in_two_image(image_1_des, image_2_des):
    
    # create BFMatcher object
    bf = cv.BFMatcher(cv.NORM_L2, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(image_1_des, image_2_des)
    
    # Sort them in the order of their distance.
    matches = sorted(matches, key = lambda x:x.distance)

    return matches