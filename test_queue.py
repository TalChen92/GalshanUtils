from yuv_queue import YUV_Queue
import numpy as np
import matplotlib.pyplot as plt
import cv2

file_path = r'C:\Users\user\Data\0.rgb'
yuv = YUV_Queue(file_path)
rgb = np.zeros((yuv.height, yuv.width, 3), dtype=np.uint16)
plt.ion()
for i in range(8):
    r_data_curr, g_data_curr, b_data_curr = yuv.read_frame(i, output='rgb_linear')
    rgb[:,:,0] += r_data_curr.astype(np.uint16)
    rgb[:,:,1] += g_data_curr.astype(np.uint16)
    rgb[:,:,2] += b_data_curr.astype(np.uint16)

inverse_glut = np.zeros(256, dtype=np.uint8)
for i in range(256):
    closest_indices = np.where(yuv.glut >= i)[0]
    if len(closest_indices) > 0:
        closest_index = closest_indices[0]
        inverse_glut[i] = closest_index

rgb = np.clip(rgb, 0, 255).astype(np.uint8)
rgb = inverse_glut[rgb]
plt.figure()
plt.imshow(rgb)
plt.title('Sum of 8 frames')
plt.show()
plt.pause(0.1)


idx = 5
thr = 48
sat = 255
kuv = 5.7

y, u, v = yuv.read_frame(idx, output='yuv')
r, g, b = yuv.read_frame(idx, output='rgb')


v_Tag = v.astype(np.int16) - 128
u_Tag = u.astype(np.int16) - 128
colot_test = r.astype(np.int16) - kuv * (v_Tag - 1.1784 * u_Tag)
r_data_curr, g_data_curr, b_data_curr = yuv.read_res_image(idx, output='rgb')
np.clip(r_data_curr, 0, 255, out=r_data_curr)
np.clip(g_data_curr, 0, 255, out=g_data_curr)   
np.clip(b_data_curr, 0, 255, out=b_data_curr)
a=1
# Create an RGBA image from the RGB data
rgba = np.zeros((yuv.height, yuv.width, 4), dtype=np.uint16)
rgba[:, :, 0] = r_data_curr
rgba[:, :, 1] = r_data_curr
rgba[:, :, 2] = r_data_curr

# Define the opacity (alpha) value
alpha = 255  # 50% opacity
rgba[:, :, 3] = alpha

# Define the color to overlay (e.g., red)
overlay_color = [255, 0, 0, 10]

# Apply the overlay color to specific pixels (e.g., where colot_test is above a threshold)
threshold = 0.5
mask = colot_test > threshold
mask2 = r_data_curr > thr
overlay_color2 = [0, 255, 0, 10]

mask3 = r>sat
overlay_color3 = [0, 0, 255, 10]

rgba[mask] = rgba[mask] + overlay_color
rgba[mask2] =  rgba[mask2] + overlay_color2
rgba[mask3] =  rgba[mask3] + overlay_color3

rgba = np.clip(rgba, 0, 255)
# Display the image with the overlay
plt.figure()
plt.imshow(rgba.astype(np.uint8))
plt.title('Frame with Overlay')
plt.show()

       