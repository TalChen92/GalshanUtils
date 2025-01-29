import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

file_name = r'rec\yuv-65.jpg'
with open(file_name,'rb') as f:
    data = f.read()

height_full = 2160
width_full = 3840

height_roi = 540
width_roi = 1920

y_full = np.frombuffer(data[:height_full*width_full], np.uint8).reshape(height_full, width_full)
y_roi = np.frombuffer(data[height_full*width_full:], np.uint8).reshape(height_roi, width_roi)

cv2.rectangle(y_full, (0,0),(540,1920))