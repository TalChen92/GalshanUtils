from yuv_queue import YUV_Queue
import numpy as np
import matplotlib.pyplot as plt

file_path = r'C:\Users\user\Documents\1.rgb'
yuv = YUV_Queue(file_path)
r_data_curr, g_data_curr, b_data_curr = yuv.read_frame(5, output='rgb')
r_data_prev, g_data_prev, b_data_prev = yuv.read_frame(2, output='rgb')
r_res = r_data_curr - r_data_prev
print(r_res[30:35,30:35])