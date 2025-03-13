from yuv_file import YUV_File
import matplotlib.pyplot as plt
import numpy as np

def plot_square(x, y, size=15, color='red'):
    s = size//2
    plt.plot([x-s, x-s, x+s, x+s, x-s], [y-s, y+s, y+s, y-s, y-s], color=color)

path = r'C:\Users\user\Data\iso500_11.36.13.674.rgb'

yuv = YUV_File(path, y_width=4080, y_height=960, y_stride=1, chroma_width=2040, chroma_height=480)
yuv.read_file()
r_data, g_data, b_data, r_data_linear, g_data_linear, b_data_linear = yuv.yuv_to_rgb()
rgb = np.zeros((480, 2040, 3), dtype=np.uint8)
rgb[:,:,0] = r_data
rgb[:,:,1] = g_data
rgb[:,:,2] = b_data
plt.figure()
plt.imshow(r_data)
plot_square(1576, 225)
plt.show()