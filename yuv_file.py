import numpy as np
import matplotlib.pyplot as plt
import cv2 
import os

class YUV_File:
    def __init__(self, file_path, y_width, y_height, y_stride, chroma_width, chroma_height):
        self.file_path = file_path
        self.y_width = y_width
        self.y_height = y_height
        #self.y_stride = y_stride
        self.chroma_width = chroma_width
        self.chroma_height = chroma_height
        #self.chroma_stride = chroma_stride
        #self.chroma_pixel_stride = chroma_pixel_stride
        self.raw_data = None
        self.y_data_full = None
        self.y_data = None
        self.u_data = None
        self.v_data = None
        self.r_data = None
        self.g_data = None
        self.b_data = None
        self.r_data_linear = None
        self.g_data_linear = None
        self.b_data_linear = None
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
    
    def read_file(self):
        with open(self.file_path, "rb") as file:
            self.raw_data = file.read()

        y_size = self.y_width * self.y_height
        u_size = self.chroma_width * self.chroma_height
        v_size = self.chroma_width * self.chroma_height
        if len(self.raw_data) != y_size + u_size + v_size:
            raise ValueError("Invalid file size")
        self.y_data_full = np.frombuffer(self.raw_data[:y_size], dtype=np.uint8).reshape((self.y_height, self.y_width))
        self.u_data = np.frombuffer(self.raw_data[y_size:y_size + u_size], dtype=np.uint8).reshape(
            (self.chroma_height, self.chroma_width))
        self.v_data = np.frombuffer(self.raw_data[y_size + u_size:], dtype=np.uint8).reshape(
            (self.chroma_height, self.chroma_width))
        self.downsample_y()
        self.r_data, self.g_data, self.b_data, self.r_data_linear, self.g_data_linear, self.b_data_linear  = self.yuv_to_rgb()

        

    def downsample_y(self):
        reshapeY = self.y_data_full.reshape((self.y_data_full.shape[0] // 2, 2, self.y_data_full.shape[1] // 2, 2))
        resizeY = reshapeY.mean(axis = (1,3)).round()
        self.y_data = resizeY.astype('uint8')

    def get_frame(self):
        file_name = os.path.basename(self.file_path)
        frame = file_name.split('_', 2)[1].split('.')[0]
        return int(frame)
    
    def extract_time_from_file_name(self):
        file_name = os.path.basename(self.file_path)
        date_str = file_name.split('_', 2)[2].split('.')[0]
        hour = int(date_str.split('_')[0])
        minute = int(date_str.split('_')[1])
        second = int(date_str.split('_')[2])
        millisecond = int(date_str.split('_')[3])
        total_ms = (hour * 60*60*1000) + (minute * 60*1000) + (second * 1000) + millisecond
        return np.uint32(total_ms)

    def get_yuv_data(self):
        return self.y_data, self.u_data, self.v_data
    
    def get_yuv_size(self):
        return self.y_data_full.size + self.u_data.size + self.v_data.size
    
    def imshow(self, type):
        if type == 'y':
            plt.figure()
            plt.imshow(self.y_data)
            plt.title('y')
            #plt.show()
        elif type == 'u':
            plt.figure()
            plt.imshow(self.u_data)
            plt.title('u')
            #plt.show()
        elif type == 'v':
            plt.figure()
            plt.imshow(self.v_data)
            plt.title('v')
            #plt.show()
        elif type == 'yuv':
            plt.figure()
            plt.imshow(self.y_data)
            plt.title('y')
            plt.figure()
            plt.imshow(self.u_data)
            plt.title('u')
            plt.figure()
            plt.imshow(self.v_data)
            plt.title('v')
            #plt.show()
        elif type == 'r':
            plt.figure()
            plt.imshow(self.r_data)
            plt.title('r')
            #plt.show()
        elif type == 'g':
            plt.figure()
            plt.imshow(self.g_data)
            plt.title('g')
            #plt.show()
        elif type == 'b':
            plt.figure()
            plt.imshow(self.b_data)
            plt.title('b')
            #plt.show()
        elif type == 'r_linear':
            plt.figure()
            plt.imshow(self.r_data_linear)
            plt.title('r_linear')
            #plt.show()
        elif type == 'g_linear':
            plt.figure()
            plt.imshow(self.g_data_linear)
            plt.title('g_linear')
            #plt.show()
        elif type == 'b_linear':
            plt.figure()
            plt.imshow(self.b_data_linear)
            plt.title('b_linear')
            #plt.show()
        elif type == 'rgb_linear':
            plt.figure()
            plt.imshow(self.r_data_linear)
            plt.title('r_linear')
            plt.figure()
            plt.imshow(self.b_data_linear)
            plt.title('g_linear')
            plt.figure()
            plt.imshow(self.g_data_linear)
            plt.title('b_linear')
            #plt.show()
        elif type == 'rgb':
            plt.figure()
            plt.imshow(self.r_data)
            plt.title('r')
            plt.figure()
            plt.imshow(self.b_data)
            plt.title('g')
            plt.figure()
            plt.imshow(self.g_data)
            plt.title('b')
            #plt.show()
        else:
            print('Invalid type - [y, u, v, yuv, rgb]')
        
    def yuv_to_rgb(self):
        r_data = self.y_data + 1.5748*(self.v_data.astype(np.float32) - 128)
        b_data = self.y_data + 1.8556*(self.u_data.astype(np.float32) - 128)
        g_data = (self.y_data - 0.2126*r_data - 0.0722*b_data)/0.7152
        r_data = np.clip(r_data,0, 255).astype(np.uint8)
        b_data = np.clip(b_data,0, 255).astype(np.uint8)
        g_data = np.clip(g_data,0, 255).astype(np.uint8)

        r_data_linear = self.glut[r_data]
        b_data_linear = self.glut[b_data]
        g_data_linear = self.glut[g_data]

        return r_data, g_data, b_data, r_data_linear, g_data_linear, b_data_linear

    def plot_histograms(self):
        r = (self.yuv_to_rgb()[:, :, 0]).flatten()
        g = (self.yuv_to_rgb()[:, :, 1]).flatten()
        b = (self.yuv_to_rgb()[:, :, 2]).flatten()

        plt.figure(figsize=(10, 6))

        plt.subplot(3, 1, 1)
        plt.hist(r, bins=256, color='red', alpha=0.7)
        plt.title('Red Channel Histogram')

        plt.subplot(3, 1, 2)
        plt.hist(g, bins=256, color='green', alpha=0.7)
        plt.title('Green Channel Histogram')

        plt.subplot(3, 1, 3)
        plt.hist(b, bins=256, color='blue', alpha=0.7)
        plt.title('Blue Channel Histogram')

        plt.tight_layout()
        plt.show()

    def save_rgb_image(self, output_path):
        rgb = self.yuv_to_rgb()
        cv2.imwrite(output_path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print(f"RGB image saved to {output_path}")
        return f"{self.make} {self.model}'s engine is running"



