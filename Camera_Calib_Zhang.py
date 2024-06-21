import math
import numpy as np
import scipy
import cv2
import os

# Read images from Calibration_Images folder and store them in a list
vis = False
images = []
path = 'Calibration_Imgs'

for filename in os.listdir(path):
    if filename.endswith('.jpg'):
        img = cv2.imread(os.path.join(path, filename))
        if img is not None:
            images.append(img)
            if vis:
                cv2.imshow('Image', img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
