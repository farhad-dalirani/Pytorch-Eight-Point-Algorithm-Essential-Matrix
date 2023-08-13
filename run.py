import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from key_feature import key_features_in_image, match_features_in_two_image
from data_util import FountainDataset
from normalized_8_point_algorithm import normalized_eight_point_essential_matrix
from calculate_epi_line import epiline_in_image_one, epiline_in_image_two
from display_util import diplay_homogeneous_line_on_image


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
    all_matches = match_features_in_two_image(image_1_des=img1_des, image_2_des=img2_des)

    # keep first 20 percent of matches
    matches = all_matches[:int(len(all_matches) * 0.2)]

    # matched points
    matched_points_1 = []
    matched_points_2 = []
    for match in all_matches[0:8]:
        matched_points_1.append([img1_key[match.queryIdx].pt[0], img1_key[match.queryIdx].pt[1]])
        matched_points_2.append([img2_key[match.trainIdx].pt[0], img2_key[match.trainIdx].pt[1]])
    matched_points_1 = np.array(matched_points_1)
    matched_points_2 = np.array(matched_points_2)

    # find essential matrix by 8 point algorithm
    results = normalized_eight_point_essential_matrix(
                    img1_points=matched_points_1,
                    img2_points=matched_points_2,
                    camera_1_matrix=camera_params_1['intrinsic_matrix'],
                    camera_2_matrix=camera_params_2['intrinsic_matrix'],
                    device='cpu')
    print("Essential Matrix:\n{}".format(results["essential_matrix"]))
    print("Epipole in image 1:\n{}".format(results["epipole_img_1"]))
    print("Epipole in image 2:\n{}".format(results["epipole_img_2"])) 


    # select 8 corresponding points for showing result of finding essential matrix
    img1_corr_points = matched_points_1[0:8]
    img2_corr_points = matched_points_2[0:8]

    # epipolar line of for some of the correspoinding points in the both image
    epiline_in_img2 = epiline_in_image_two(
                            essential_matrix=results["essential_matrix"],
                            image_points_in_image_1=img1_corr_points,
                            camera_matirx_1=camera_params_1['intrinsic_matrix'],
                            camera_matirx_2=camera_params_2['intrinsic_matrix'])

    epiline_in_img1 = epiline_in_image_one(
                            essential_matrix=results["essential_matrix"],
                            image_points_in_image_2=img2_corr_points,
                            camera_matirx_1=camera_params_1['intrinsic_matrix'],
                            camera_matirx_2=camera_params_2['intrinsic_matrix'])                     

    # draw epipolar lines for a pair of corresponding points in the both images
    image_1_epiline = diplay_homogeneous_line_on_image(image=image_1, lines_hm=epiline_in_img1, points=img1_corr_points)
    image_2_epiline = diplay_homogeneous_line_on_image(image=image_2, lines_hm=epiline_in_img2, points=img2_corr_points)
    img_1_2_epiline = np.concatenate((image_1_epiline, image_2_epiline), axis=1)

    # draw matches
    img3 = cv.drawMatches(image_1, img1_key, image_2, img2_key, matches, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
   
    plt.figure()
    plt.imshow(keypoint_on_image_1)
    plt.title('Key features in the first image')
    plt.figure()
    plt.imshow(keypoint_on_image_2)
    plt.title('Key features in the second image')
    plt.figure()
    plt.imshow(img3)
    plt.title('Some of top best matched key features')
    plt.figure()
    plt.imshow(img_1_2_epiline)
    plt.title('Epipolar lines on images for some corresponding key feature pairs.')
    plt.show()