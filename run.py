import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from key_feature import key_features_in_image, match_features_in_two_image
from data_util import FountainDataset

if __name__ == '__main__':
    
    # read input images
    dataset = FountainDataset(root_path='./data/')
    camera_params_1 = dataset.read_camera_parameters(pose_number=5)
    image_1 = dataset.read_image(pose_number=5)
    camera_params_2 = dataset.read_camera_parameters(pose_number=6)
    image_2 = dataset.read_image(pose_number=6)

    # features in image 1 and 2
    img1_key, img1_des = key_features_in_image(image=image_1)
    img2_key, img2_des = key_features_in_image(image=image_2)
    
    # drawing the keypoints
    keypoint_on_image_1 = cv.drawKeypoints(image_1, img1_key, None, color=(0, 255, 0), flags=0)
    keypoint_on_image_2 = cv.drawKeypoints(image_2, img2_key, None, color=(0, 255, 0), flags=0)

    # match key featues in both images
    matches = match_features_in_two_image(image_1_des=img1_des, image_2_des=img2_des)

    # keep first 20 percent of matches
    matches = matches[:int(len(matches)*0.2)]

    # draw matches
    img3 = cv.drawMatches(image_1, img1_key, image_2, img2_key, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    plt.figure()
    plt.imshow(img3),plt.show()

    plt.figure()
    plt.imshow(image_1)
    plt.figure()
    plt.imshow(image_2)
    plt.figure()
    plt.imshow(keypoint_on_image_1)
    plt.figure()
    plt.imshow(keypoint_on_image_2)
    plt.show()