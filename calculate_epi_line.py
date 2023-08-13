import numpy as np

def epiline_in_image_one(essential_matrix, image_points_in_image_2, camera_matirx_1, camera_matirx_2):
    """
    Calculate epipolar lines in camera one for a points in camera two.

    essential_matrix: numpy array 3 * 3
    image_point_in_image_1: pixel coordinate in shape (N, 2)
    camera_matirx_1: camera matrix in shape 3 * 3
    """
    # convert point to homogeneous
    img2_points = np.concatenate((image_points_in_image_2.T, np.ones(shape=(1, image_points_in_image_2.shape[0]))), axis=0) # 3 * N
    
    # find ray that passess from origin of camera coordinate thorough point
    points_ldr = np.dot(np.linalg.inv(camera_matirx_2), img2_points) # 3 * N

    # find epiline in the other image camera in homogeneous coordinate
    lines = np.dot(essential_matrix.T, points_ldr) # 3 * N
    epilines = np.dot(np.linalg.inv(camera_matirx_1.T), lines)

    return epilines


def epiline_in_image_two(essential_matrix, image_points_in_image_1, camera_matirx_1, camera_matirx_2):
    """
    Calculate epipolar lines in camera one for a points in camera two.
    
    essential_matrix: numpy array 3 * 3
    image_point_in_image_1: pixel coordinate in shape (N, 2)
    camera_matirx_1: camera matrix in shape 3 * 3
    """
    # convert point to homogeneous
    img1_points = np.concatenate((image_points_in_image_1.T, np.ones(shape=(1, image_points_in_image_1.shape[0]))), axis=0) # 3 * N
    
    # find ray that passess from origin of camera coordinate thorough point
    points_ldr = np.dot(np.linalg.inv(camera_matirx_1), img1_points) # 3 * N

    # find epiline in the other image camera in homogeneous coordinate
    lines = np.dot(essential_matrix, points_ldr) # 3 * N
    epipolar_lines = np.dot(np.linalg.inv(camera_matirx_2.T), lines) # 3 * N

    return epipolar_lines
