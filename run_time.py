import torch
import time
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from key_feature import key_features_in_image, match_features_in_two_image
from data_util import FountainDataset
from eight_point_algorithm import eight_point_essential_matrix
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
    matches = all_matches[:int(len(all_matches) * 1)]
    print("Number of kepted matches: {}".format(len(matches)))

    # time
    execution_time_ms_cpu = []
    execution_time_ms_gpu = []
    execution_time_ms_ran = []
    execution_time_ms_lme = []

    # Dummy operation to warm up the GPU
    dummy_input = torch.zeros(1).cuda()

    for num_points in range(8, 2400):
        # matched points
        matched_points_1 = []
        matched_points_2 = []
        for match in all_matches[0:num_points]:
            matched_points_1.append([img1_key[match.queryIdx].pt[0], img1_key[match.queryIdx].pt[1]])
            matched_points_2.append([img2_key[match.trainIdx].pt[0], img2_key[match.trainIdx].pt[1]])
        matched_points_1 = np.array(matched_points_1)
        matched_points_2 = np.array(matched_points_2)

        start_time_cpu = time.time()
        # find essential matrix by 8 point algorithm
        results_cpu = eight_point_essential_matrix(
                        img1_points=matched_points_1,
                        img2_points=matched_points_2,
                        camera_1_matrix=camera_params_1['intrinsic_matrix'],
                        camera_2_matrix=camera_params_2['intrinsic_matrix'],
                        device='cpu')
        end_time_cpu = time.time()
        
        start_time_gpu = time.time()
        # find essential matrix by 8 point algorithm
        results_gpu = eight_point_essential_matrix(
                        img1_points=matched_points_1,
                        img2_points=matched_points_2,
                        camera_1_matrix=camera_params_1['intrinsic_matrix'],
                        camera_2_matrix=camera_params_2['intrinsic_matrix'],
                        device='cuda')
        end_time_gpu = time.time()

        start_time_ran = time.time()
        essential_matrix_1, _ = cv.findEssentialMat(
                            matched_points_1,
                            matched_points_2,
                            cameraMatrix=camera_params_1['intrinsic_matrix'],
                            method=cv.RANSAC, prob=0.999, threshold=1.0, mask=None)
        end_time_ran = time.time()

        start_time_lme = time.time()
        essential_matrix_2, _ = cv.findEssentialMat(
                            matched_points_1,
                            matched_points_2,
                            cameraMatrix=camera_params_1['intrinsic_matrix'],
                            method=cv.LMEDS)
        end_time_lme = time.time()

        # Calculate the execution time in milliseconds
        execution_time_ms_cpu.append((end_time_cpu - start_time_cpu) * 1000)
        execution_time_ms_gpu.append((end_time_gpu - start_time_gpu) * 1000)
        execution_time_ms_ran.append((end_time_ran - start_time_ran) * 1000)
        execution_time_ms_lme.append((end_time_lme - start_time_lme) * 1000)
        
        print("=" * 100)
        print('Number of points:{}'.format(num_points))
        print("Essential Matrix:\n{}".format(results_cpu["essential_matrix"]))
        print("Essential Matrix:\n{}".format(results_gpu["essential_matrix"]))
        print("Opencv Essential Matrix (RANSAC):\n{}".format(essential_matrix_1))
        print("Opencv Essential Matrix (LMedS):\n{}".format(essential_matrix_2))
    

    num_p = len(execution_time_ms_cpu)
    print("PyTorch-CPU: {}\nPyTorch-GPU: {}\nOpenCV-ran: {}\nOpenCV-lme: {}\n".format(
        sum(execution_time_ms_cpu)/num_p, 
        sum(execution_time_ms_gpu)/num_p, 
        sum(execution_time_ms_ran)/num_p, 
        sum(execution_time_ms_lme)/num_p))
    plt.plot([i for i in range(8, num_p+8)], execution_time_ms_cpu, label='pytorch-cpu')
    plt.plot([i for i in range(8, num_p+8)], execution_time_ms_gpu, label='pytorch-gpu')
    plt.plot([i for i in range(8, num_p+8)], execution_time_ms_ran, label='OpenCV-ran')
    plt.plot([i for i in range(8, num_p+8)], execution_time_ms_lme, label='OpenCV-lme')
    plt.xlabel('Number of corresponding points')
    plt.ylabel('Millisecond')
    plt.legend()

    plt.plot([i for i in range(8, num_p+8)], execution_time_ms_cpu, label='pytorch-cpu')
    plt.plot([i for i in range(8, num_p+8)], execution_time_ms_gpu, label='pytorch-gpu')
    plt.plot([i for i in range(8, num_p+8)], execution_time_ms_lme, label='OpenCV-lme')
    plt.xlabel('Number of corresponding points')
    plt.ylabel('Millisecond')
    plt.legend()

    plt.show()