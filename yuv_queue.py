import numpy as np
import matplotlib.pyplot as plt
import cv2 
import os

class YUV_Queue:
    def __init__(self, file_path):
        self.file_path = file_path
        self.glut = np.array([0,0,0,1,1,1,1,2,2,2,2,2,3,3,3,3,
        4,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,
        7,8,8,8,9,9,9,10,10,10,11,11,11,12,12,13,
        13,13,14,14,15,15,15,16,16,17,17,18,18,19,19,20,
        20,21,21,22,22,23,23,24,24,25,25,26,27,27,28,28,
        29,30,30,31,31,32,33,33,34,35,35,36,37,37,38,39,
        40,40,41,42,43,43,44,45,46,46,47,48,49,50,51,51,
        52,53,54,55,56,57,57,58,59,60,61,62,63,64,65,66,
        67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,
        83,84,85,86,88,89,90,91,92,93,94,96,97,98,99,100,
        102,103,104,105,107,108,109,110,112,113,114,116,117,118,119,121,
        122,124,125,126,128,129,130,132,133,135,136,138,139,140,142,143,
        145,146,148,149,151,152,154,155,157,158,160,162,163,165,166,168,
        170,171,173,174,176,178,179,181,183,184,186,188,190,191,193,195,
        197,198,200,202,204,205,207,209,211,213,214,216,218,220,222,224,
        226,228,229,231,233,235,237,239,241,243,245,247,249,251,253,255])
        self.read_header()
    
    def read_header(self, header_size=64):
        with open(self.file_path, "rb") as file:
            self.raw_data = file.read(header_size)
        self.first_frame = np.frombuffer(self.raw_data[0:4], dtype=np.uint32)[0]
        self.last_frame = np.frombuffer(self.raw_data[4:8], dtype=np.uint32)[0]
        self.height = np.frombuffer(self.raw_data[8:12], dtype=np.uint32)[0]
        self.width = np.frombuffer(self.raw_data[12:16], dtype=np.uint32)[0]
        self.detection_frame = np.frombuffer(self.raw_data[16:20], dtype=np.uint32)[0]
        self.detection_x = np.frombuffer(self.raw_data[20:24], dtype=np.uint32)[0]
        self.detection_y = np.frombuffer(self.raw_data[24:28], dtype=np.uint32)[0]
        self.padding = np.frombuffer(self.raw_data[28:64], dtype=np.uint32)

    def read_frame(self, frame_number, output = 'yuv', abs=False):
        if abs:
            frame_number = frame_number - self.first_frame
        if frame_number < 0 or frame_number > self.last_frame - self.first_frame:
            raise ValueError("Invalid frame number")
        with open(self.file_path, "rb") as file:
            file.seek(64 + frame_number * self.width * self.height * 3)
            raw_data = file.read(self.width * self.height * 3)
        y_data = np.frombuffer(raw_data[:self.width * self.height], dtype=np.uint8).reshape((self.height, self.width))
        u_data = np.frombuffer(raw_data[self.width * self.height : self.width * self.height * 2], dtype=np.uint8).reshape((self.height, self.width))
        v_data = np.frombuffer(raw_data[self.width * self.height * 2 :], dtype=np.uint8).reshape((self.height, self.width))
        if output == 'yuv':
            return y_data, u_data, v_data
        elif output == 'rgb':
            return self.yuv_to_rgb(y_data, u_data, v_data)
        elif output == 'rgb_linear':
            return self.yuv_to_rgb_linear(y_data, u_data, v_data)
        else:
            raise ValueError("Invalid output format")
        
    def yuv_to_rgb(self, y_data, u_data, v_data):
        r_data = y_data + 1.5748*(v_data.astype(np.float32) - 128)
        b_data = y_data + 1.8556*(u_data.astype(np.float32) - 128)
        g_data = (y_data - 0.2126*r_data - 0.0722*b_data)/0.7152

        r_data = np.clip(r_data,0, 255).astype(np.uint8)
        b_data = np.clip(b_data,0, 255).astype(np.uint8)
        g_data = np.clip(g_data,0, 255).astype(np.uint8)

        return r_data, g_data, b_data
    
    def yuv_to_rgb_linear(self, y_data, u_data, v_data):
        r_data, g_data, b_data = self.yuv_to_rgb(y_data, u_data, v_data)
        return self.glut[r_data], self.glut[g_data], self.glut[b_data]

    def read_res_image(self, frame_number, output = 'yuv', abs=False, diff=3):
        curr_1, curr_2, curr_3 = self.read_frame(frame_number, output, abs)
        prev_1, prev_2, prev_3 = self.read_frame(frame_number - diff, output, abs)
        res_1 = curr_1.astype(np.int32) - prev_1.astype(np.int32)
        res_2 = curr_2.astype(np.int32) - prev_2.astype(np.int32)
        res_3 = curr_3.astype(np.int32) - prev_3.astype(np.int32)
        return res_1, res_2, res_3