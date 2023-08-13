import numpy as np
import cv2 as cv

def diplay_homogeneous_line_on_image(image, lines_hm, points=None, radious=20):
    """
    image: a numpy array in shape of N*M
    lines_hm: it is a homogeneous line in format of (a, b, c)
    """
    # create a copy of image for drawing inside it
    image_lines = np.copy(image)
    
    # for each homogeneous line
    lines_hm = lines_hm.T
    for i_th, line in enumerate(lines_hm):
        a, b, c = line
        x0 = 0
        x1 = image.shape[1]
        y0 = int(-c / b)
        y1 = int(-(a * x1 + c) / b)
        
        # Draw the epiline on the image
        color_i = color(i_th)
        cv.line(image_lines, (x0, y0), (x1, y1), color_i, 3)
    
        if points is not None:
            cv.circle(image_lines, tuple([int(points[i_th][0]), int(points[i_th][1])]), radious, color_i, -1)  

    return image_lines

def color(i):
    colors = [
        (255, 0, 0),     # Red
        (0, 255, 0),     # Green
        (0, 0, 255),     # Blue
        (255, 255, 0),   # Yellow
        (0, 255, 255),   # Cyan
        (255, 0, 255),   # Magenta
        (128, 0, 0),     # Dark Red
        (0, 128, 0),     # Dark Green
        (0, 0, 128),     # Dark Blue
        (128, 128, 0),   # Olive
        (128, 0, 128),   # Purple
        (0, 128, 128),   # Teal
        (255, 128, 0),   # Orange
        (128, 255, 0),   # Lime Green
        (0, 128, 255),   # Sky Blue
        (255, 0, 128),   # Pink
        (0, 255, 128),   # Spring Green
        (128, 0, 255),   # Violet
        (255, 128, 128), # Light Salmon
        (128, 255, 128)  # Light Green
    ]

    return colors[i % len(colors)]