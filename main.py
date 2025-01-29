from yuv_file import YUV_File
from yuv_movie import YUV_Movie
from yuv_movie import calc_color_statistic
import matplotlib.pyplot as plt
import json
import cv2
import numpy as np

mode = 1
if mode == 0: # test movie
    file_path = r'..\data\results\movie_0.yuv1'
    yuv = YUV_Movie(file_path)
    print(yuv)
    print(f"Average fps: {yuv.calc_average_fps()}")
    rgb = yuv.read_rgb_frame(10)
    line = 100
    col = 200
    win_size = 11
    stats = calc_color_statistic(rgb, line, col, win_size)
    plt.figure()
    plt.imshow(rgb)
    plt.show()
    print(f"Color statistic on window\nCenter: {line}, {col}\nWindow size: {win_size}\n {json.dumps(stats, indent=4)}")
    #y, u, v, timestamp = yuv.read_frame(0)
    #yuv.play_movie(0, 50)



elif mode == 1: # test_file
    curr_file_path = r'2025.01.13_16.18.54\16.18.54.410.rgb'
    prev_file_path = r'2025.01.13_16.18.54\16.18.54.318.rgb'
    galaxyModel = 24#24
    roi = True
    if galaxyModel == 22:
        y_row_stride = 4096
        chroma_stride = 2048#4096
    elif galaxyModel == 21:
        y_row_stride = 4000
        chroma_stride = 2000#4096
    elif galaxyModel == 24:
        y_row_stride = 4080
        chroma_stride = 4080
    else:
        print('Unsupported galaxyModel should be 22/24')
        exit()
    y_width = y_row_stride
    chroma_width = chroma_stride // 2
    chroma_pixel_stride = 2#2
    if roi:
        y_height = 960

    else:
        y_height = 3060
    chroma_height = y_height // 2

    curr_file = YUV_File(curr_file_path, y_width, y_height, y_row_stride, chroma_width, chroma_height)
    prev_file = YUV_File(prev_file_path, y_width, y_height, y_row_stride, chroma_width, chroma_height)
    #time = yuv.extract_time_from_file_name()
    curr_file.read_file()
    prev_file.read_file()
    #curr_file.imshow('yuv')
    res = np.clip(curr_file.r_data.astype(np.float32) - prev_file.r_data,0,255).astype(np.uint8)
    drthr_tuple = cv2.threshold(res,16,255,cv2.THRESH_BINARY)
    #print(yuv.get_yuv_size())
    #curr_file.imshow(type='r')
    #curr_file.imshow(type='r_linear')
    #prev_file.imshow(type='r')
    #prev_file.imshow(type='r_linear')
    drthr=drthr_tuple[1]

    sat =cv2.threshold(curr_file.r_data,250,255,cv2.THRESH_BINARY_INV)
    drthr1 = cv2.bitwise_and(sat[1],drthr)
    kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    kernel_big = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))

    a=cv2.morphologyEx(drthr1, cv2.MORPH_ERODE, kernel_small)
    b=cv2.morphologyEx(a, cv2.MORPH_DILATE, kernel_big)
    c= np.clip(drthr1.astype(np.float32)-b,0,255).astype(np.uint8)

    plt.figure()
    plt.imshow(drthr1)
    plt.figure()
    plt.imshow(a)
    plt.figure()
    plt.imshow(b)
    plt.figure()
    plt.imshow(c)
    plt.show()
