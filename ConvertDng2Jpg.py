import rawpy
import imageio
import numpy as np
from PIL import Image
import inspect
import exifread
import exiftool

XYZ_to_sRGB = np.array([
    [3.2406, -1.5372, -0.4986],
    [-0.9689, 1.8758, 0.0415],
    [0.0557, -0.2040, 1.0570]
])

def get_camera_to_xyz_matrix(raw, color_temperature=6000):
    
    color_matrix1 = raw.color_matrix
    if hasattr(raw, 'color_matrix_d65'):
        color_matrix2 = raw.color_matrix_d65
        temperature1 = 2850
        temperature2 = 6500
        interpolation_factor = (color_temperature - temperature1)/(temperature2 - temperature1)
        camera_to_xyz = color_matrix1 * (1 - interpolation_factor) + color_matrix2 * interpolation_factor
    else:
        camera_to_xyz = color_matrix1[:3,:3]

    return camera_to_xyz.reshape(3, 3)

def gamma_correction(channel, gamma=2.2):
    return np.where(channel <= 0.0031308, 12.92 * channel, 1.055 * (channel ** (1 / gamma)) -0.055)

def conveert_to_sRGB(raw_image, camera_to_xyz):
    raw_data = raw_image.raw_image_visible.astype(np.float32)
    wb_multipliers = raw_image.camera_whitebalance
    raw_data[..., 0] *= wb_multipliers[0]
    raw_data[..., 1] *= wb_multipliers[1]
    raw_data[..., 2] *= wb_multipliers[2]
    raw_data /= np.max(raw_data)
    height, width = raw_data.shape
    raw_data = raw_data.reshape(-1, 3)
    xyz_data = np.dot(raw_data, camera_to_xyz.T)
    srgb_data = np.dot(xyz_data, XYZ_to_sRGB.T)
    srgb_data = np.clip(srgb_data, 0, 1)
    srgb_data = gamma_correction(srgb_data)
    srgb_image = (srgb_data * 255).astype(np.uint8)
    srgb_image = srgb_image.reshape(height, width, 3)
    return srgb_image

def print_raw_data(raw):
    raw_attributes = dir(raw)
    public_attributes = [attr for attr in raw_attributes if not attr.startswith('_')]
    for attr in public_attributes:
        try:
            value = getattr(raw, attr)
            if inspect.ismethod(value) or inspect.isfunction(value):
                continue
            print(f'{attr}: {value}')
        except:
            print(f'Unable to print {attr}')


dng_file = '20240709_163330.dng'
with exiftool.ExifTool() as et:
    metadata = et.get_metadata(dng_file)

with open(dng_file, 'rb') as f:
    tags = exifread.process_file(f)

for tag in tags:
    print(f'{tag}: {tags[tag]}')

with rawpy.imread(dng_file) as raw:
    print_raw_data(raw)
    rgb_img = raw.postprocess()
    camera_to_xyz = get_camera_to_xyz_matrix(raw)
    srgb_image = conveert_to_sRGB(raw, camera_to_xyz)

image = Image.fromarray(srgb_image)
image.save('output_image1.jpg','JPEG')