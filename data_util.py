import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt

class FountainDataset:

    def __init__(self, root_path):
        self.root_path = root_path
        self.number_poses = 10

    def read_image(self, pose_number):

        file_path = os.path.join(self.root_path, 'fountain_dense_images', "{}.png".format(str(pose_number).zfill(4)))

        image = cv.imread(filename=file_path)
        image = cv.cvtColor(src=image, code=cv.COLOR_BGR2RGB)

        return image

    def read_camera_parameters(self, pose_number):

        file_path = os.path.join(self.root_path, 'fountain_dense_cameras', "{}.png.camera".format(str(pose_number).zfill(4)))

        # open file containing camera parameters
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # Read intrinsic matrix (3x3)
        intrinsic_matrix = np.array([list(map(float, lines[i].split())) for i in range(3)])

        # Read sensor resolution (2 elements)
        resolution = tuple(map(int, lines[8].split()))

        return {
            'intrinsic_matrix': intrinsic_matrix,
            'resolution': resolution
        }

if __name__ == '__main__':
    
    # Example usage
    dataset = FountainDataset(root_path='./data/')
    
    camera_params = dataset.read_camera_parameters(pose_number=1)
    print(camera_params)

    img = dataset.read_image(pose_number=1)
    plt.imshow(img)
    plt.show()
