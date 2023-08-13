import numpy as np
import cv2 as cv

def diplay_homogeneous_line_on_image(image, lines_hm, point=None, radious=20):
    """
    image: a numpy array in shape of N*M
    lines_hm: it is a homogeneous line in format of (a, b, c)
    """
    # create a copy of image for drawing inside it
    image_lines = np.copy(image)
    
    # for each homogeneous line
    lines_hm = lines_hm.reshape(-1, 3)
    for line in lines_hm:
        a, b, c = line
        x0 = 0
        x1 = image.shape[1]
        y0 = int(-c / b)
        y1 = int(-(a * x1 + c) / b)
        
        # Draw the epiline on the image
        cv.line(image_lines, (x0, y0), (x1, y1), (0, 255, 0), 3)
    
    if point is not None:
        cv.circle(image_lines, tuple([int(point[0]), int(point[1])]), radious, (0, 0, 255), -1)  

    return image_lines