import cv2
import numpy as np
import os
from yuv_file import YUV_File



base_path = r'C:\Users\user\Data\G20\10-03-25\15.03.48\Recordings\2025.03.10_15.03.52'
files = os.listdir(base_path)
yuv_files = [file for file in files if file.endswith('.rgb')]
yuv_files.sort()
rgb_sum = np.zeros((480, 2040, 3), dtype=np.float32)
for yuv_file in yuv_files[:10]:
    path = os.path.join(base_path, yuv_file)
    yuv = YUV_File(path, y_width=4080, y_height=960, y_stride=1, chroma_width=2040, chroma_height=480)
    yuv.read_file()
    r_data, g_data, b_data, r_data_linear, g_data_linear, b_data_linear = yuv.yuv_to_rgb()
    rgb = np.stack((r_data, g_data, b_data), axis=-1)
    rgb_sum += rgb

rgb_clip = np.clip(rgb_sum/2, 0, 255).astype(np.uint8)

rgb_tonemapped = cv2.createTonemapReinhard(gamma=2.2).process(rgb_sum)
cv2.imshow('Tonemapped', rgb_tonemapped)
cv2.waitKey(0)