import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import zoom

def downsampleY(y):
    reshapeY = y.reshape((y.shape[0] // 2, 2, y.shape[1] // 2, 2))
    resizeY = reshapeY.mean(axis = (1,3)).round()
    return resizeY.astype('uint8')

def read_y_with_stride(file_path, width=4080,row_stride=4096, height=3060, uv_pixel_stride=2):
    
    chroma_height = height // 2
    chroma_width = width // 2
    y_data = np.zeros((height, width), dtype=np.uint8)
    u_data = np.zeros((chroma_height, chroma_width), dtype=np.uint8)
    v_data = np.zeros((chroma_height, chroma_width), dtype=np.uint8)



    with open(file_path, "rb") as f:
        for row in range(height):
            row_data = f.read(width)
            if not row_data:
                print('file ended')
            y_data[row, :] = np.frombuffer(row_data, dtype=np.uint8)
            f.seek(row_stride - width, 1)
        f.seek(12533744, 0)
        for row in range(chroma_height):
            row_data = f.read(chroma_width * uv_pixel_stride)
            if not row_data:
               print('file ended')
            u_data[row, :] = np.frombuffer(row_data[::uv_pixel_stride], dtype=np.uint8)
            f.seek(row_stride - chroma_width * uv_pixel_stride, 1)
            
        f.seek(12533744+6266863, 0)

        for row in range(chroma_height):
            row_data = f.read(chroma_width * uv_pixel_stride)
            if not row_data:
                print('file ended')
            v_data[row, :] = np.frombuffer(row_data[::uv_pixel_stride], dtype=np.uint8)
            f.seek(row_stride - chroma_width * uv_pixel_stride, 1)

    return y_data, u_data, v_data

def readYUV(yuv_path, width=3840, height=2160, pixel_stride=2):
    with open(yuv_path,"rb") as file:
        yuv422_data = file.read()
    y_size = width * height
    u_size = width * height // 2 - 1
    u_size = width * height // 2 - 1
    y = np.frombuffer(yuv422_data[:y_size],dtype=np.uint8).reshape((height, width))
    u = np.frombuffer(yuv422_data[y_size:y_size+u_size:2],dtype=np.uint8).reshape((height//2, width//2))
    v = np.frombuffer(yuv422_data[y_size+u_size::2],dtype=np.uint8).reshape((height//2, width//2))
    return y, u, v

def read_image(image_path):
    with open(image_path,"rb") as file: 
        metadata = file.read(12)
        data = file.read()
        number_of_rows = np.frombuffer(metadata[:4], np.int32)[0]
        number_of_columns = np.frombuffer(metadata[4:8], np.int32)[0]
        type = np.frombuffer(metadata[8:], np.int32)[0]
        y =  np.frombuffer(data[::3], np.uint8).reshape((number_of_rows, number_of_columns)).astype(np.float32)
        u =  np.frombuffer(data[1::3], np.uint8).reshape((number_of_rows, number_of_columns)).astype(np.float32)
        v =  np.frombuffer(data[2::3], np.uint8).reshape((number_of_rows, number_of_columns)).astype(np.float32)
        return y, u, v


def readChannel(channel_path, width=3840, height=2160):
    with open(channel_path,"rb") as file:
        channel_data = file.read()
    size = width * height
    if len(channel_data) != size:
        print('Unrecognize fille size')
        return
    out = np.frombuffer(channel_data,dtype=np.uint8).reshape((height, width))
    #out = np.frombuffer(channel_data,dtype=np.uint8).reshape((width, height))

  
    return out

def readYUV1(yuv_path, width=3840, height=2160, pixel_stride=2):
    with open(yuv_path,"rb") as file:
        yuv422_data = file.read()
    y_size = width * height
    u_size = width * height // 4 
    u_size = width * height // 4 
    y = np.frombuffer(yuv422_data[:y_size],dtype=np.uint8).reshape((height, width))
    u = np.frombuffer(yuv422_data[y_size:y_size+u_size],dtype=np.uint8).reshape((height//2, width//2))
    v = np.frombuffer(yuv422_data[y_size+u_size:],dtype=np.uint8).reshape((height//2, width//2))
    return y, u, v



file_path = r'..\data\detect_2025.01.14_16.20.38_452.rgb'
file_path_r = r'..\data\r20.rgb'


lut = [0,0,0,1,1,1,1,2,2,2,2,2,3,3,3,3,
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
        226,228,229,231,233,235,237,239,241,243,245,247,249,251,253,255]
y, u, v = read_image(file_path)
r = y + (v-128)*1.5748
b = y + 1.8556*(u-128)
g = (y - 0.2126*r -0.0722*b)/0.7152
plt.figure()
plt.imshow(r)
plt.show()

line = 277
column = 1129
r_linear = lut[int(r[line][column])]
g_linear = lut[int(g[line][column])]
b_linear = lut[int(b[line][column])]


print((y[line][column], u[line][column], v[line][column]))

print((r_linear, g_linear, b_linear))
a=1
'''
file_path_0 = r'..\data\5.34.33.626.rgb'
width = 2000
height = 382
with open(file_path_0,"rb") as file:
    yuv422_data = file.read()
y_size = width * height
u_size = width * height 
u_size = width * height 
y = np.frombuffer(yuv422_data[:y_size],dtype=np.uint8).reshape((height, width))
u = np.frombuffer(yuv422_data[y_size:y_size+u_size],dtype=np.uint8).reshape((height, width))
v = np.frombuffer(yuv422_data[y_size+u_size:],dtype=np.uint8).reshape((height, width))
yuv = np.stack((y, u, v), axis=-1)
rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
plt.figure()
plt.imshow(y)
plt.title('y')
plt.figure()
plt.imshow(u)
plt.title('u')
plt.figure()
plt.imshow(v)
plt.title('v')
plt.figure()
plt.imshow(rgb)
plt.title('rgb')
plt.show()
'''
'''
yuv=[]
file_path_0 = r'..\data\detect_2024.07.18_17.04.02_390.rgb'
y_list = []
u_list = []
v_list = []
r_list = []

with open(file_path_0,"rb") as file:
    for i in range(5):
        width = 2040
        height = 480
        y_data = file.read(width * height)  
        y = np.frombuffer(y_data,dtype=np.uint8).reshape((height, width))
        u_data = file.read(width * height)  
        u = np.frombuffer(u_data,dtype=np.uint8).reshape((height, width))
        v_data = file.read(width * height)  
        v = np.frombuffer(v_data,dtype=np.uint8).reshape((height, width))
        r_data = file.read(width * height)  
        r = np.frombuffer(r_data,dtype=np.uint8).reshape((height, width))
        y_list.append(y)
        u_list.append(u)
        v_list.append(v)
        r_list.append(r)

plt.show()
'''
file_path_0 = r'..\data\detect_2024.07.18_17.04.02_390.rgb'
width = 2040
height = 480
with open(file_path_0,"rb") as file:
    temp = file.read(12)
    yuv422_data = file.read()
y_size = width * height
u_size = width * height 
v_size = width * height 
y = np.frombuffer(yuv422_data[:y_size],dtype=np.uint8).reshape((height, width))
u = np.frombuffer(yuv422_data[y_size:y_size+u_size],dtype=np.uint8).reshape((height, width))
v = np.frombuffer(yuv422_data[y_size+u_size:],dtype=np.uint8).reshape((height, width))
yuv = np.stack((y, u, v), axis=-1)
rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
plt.figure()
plt.imshow(rgb)
plt.title('rgb')
plt.show()

y_0, u_0, v0 = read_y_with_stride(file_path_0)
plt.figure()
plt.imshow(y_0)
plt.title('y-0')

file_path_01 = r'C:\Users\user\Documents\Tests\f0\yuv_10.jpg'
y_01, u_01, v01 = read_y_with_stride(file_path_01)
plt.figure()
plt.imshow(y_01)
plt.title('y-01')

file_path_1 = r'C:\Users\user\Documents\Tests\f0\yuv_10.jpg'
y_1, u_1, v1 = read_y_with_stride(file_path_0)
plt.figure()
plt.imshow(y_1)
plt.title('y-1')
plt.show()