import numpy as np

def epiline_in_image_one(essential_matrix, image_point_in_image_2, camera_matirx_1, camera_matirx_2):

    # convert point to homogeneous
    img2_point = image_point_in_image_2.reshape((2, 1)) # 2*1
    img2_point = np.concatenate((img2_point, np.ones(shape=(1,1))), axis=0) # 3*1
    
    # find ray that passess from origin of camera coordinate thorough point
    point_ldr = np.dot(np.linalg.inv(camera_matirx_2), img2_point) # 3*1

    # find epiline in the other image camera in homogeneous coordinate
    line = np.dot(essential_matrix.T, point_ldr) # 3 * 1
    epiline = np.dot(np.linalg.inv(camera_matirx_1.T), line)

    return epiline


def epiline_in_image_two(essential_matrix, image_point_in_image_1, camera_matirx_1, camera_matirx_2):
    """
    essential_matrix: numpy array 3*3
    image_point_in_image_1: pixel coordinate in shape (2, )
    camera_matirx_1: camera matrix in shape 3*3
    """

    # convert point to homogeneous
    img1_point = image_point_in_image_1.reshape((2, 1))
    img1_point = np.concatenate((img1_point, np.ones(shape=(1,1))), axis=0)
    
    # find ray that passess from origin of camera coordinate thorough point
    point_ldr = np.dot(np.linalg.inv(camera_matirx_1), img1_point)

    # find epiline in the other image camera in homogeneous coordinate
    line = np.dot(essential_matrix, point_ldr)
    epiline = np.dot(np.linalg.inv(camera_matirx_2.T), line)

    return epiline
