import os
import sys
import argparse
from yuv_file import YUV_File
import re
import numpy as np


def init_file(file_path, config):
    f = open(file_path, 'wb')
    f.write(config)
    return f

def build_config(file_path, date):
    '''
    Defined Header Fields
    1.	Date
        o	Offset: 0
        o	Type: uint32
        o	Description: The capture date of the video in YYYYMMDD format.
        o	Size: 4 bytes
    2.	ISO
        o	Offset: 4
        o	Type: uint16
        o	Description: The ISO sensitivity setting of the camera.
        o	Size: 2 bytes
    3.	Shutter Time
        o	Offset: 6
        o	Type: float32
        o	Description: The shutter time in seconds.
        o	Size: 4 bytes
    4.	White Balance (WB)
        o	Offset: 10
        o	Type: uint16
        o	•  Description: The white balance setting of the camera, defined in Appendix A.
        o	•  Size: 2 bytes
        5.	Focus
        o	Offset: 12
        o	Type: uint16
        o	•  Description: Focus setting in diopters, as outlined in Appendix B.
        o	•  Size: 2 bytes

    '''
    with open(file_path, 'r') as f:
        config_str = f.read()
    config = bytearray(b'\0' * 1024)

    config[0:4] = date.tobytes()

    iso_match = sensitivity_match = re.search(r"Sensitivity\(ISO\)\s*=\s*(\d+)", config_str)
    iso = np.uint16(iso_match.group(1)) if iso_match else 0
    config[4:6] = iso.tobytes()

    shutter_match = re.search(r"Exposure\(nanoseconds\)\s*=\s*([\d.]+)", config_str)
    shutter = np.float32(shutter_match.group(1)) if shutter_match else 0
    config[6:10] = shutter.tobytes()

    wb_match = re.search(r"White Balance Mode.*?=\s*(\d+)", config_str)
    wb = np.uint16(wb_match.group(1)) if wb_match else None
    config[10:12] = wb.tobytes()


    focus_match = re.search(r"Focus\(diopter\)\s*=\s*([\d.]+)", config_str)
    focus = np.uint16(float(focus_match.group(1))*100) if focus_match else None
    config[12:14] = focus.tobytes()


    return config

def extract_date_from_folder_name(folder_name):
    match = re.match(r"(\d{4})-(\d{2})-(\d{2})", folder_name)
    if match:
        year, month, day = match.groups()
        # Create YYYYMMDD as an integer
        date_as_int = int(year + month + day)
        # Convert to uint32
        return  np.uint32(date_as_int)
    else:
        print("Invalid folder name format.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process folder with yuv files to create movies of num_files frames')
    parser.add_argument('folder_path', type=str, help='Path to the folder')
    parser.add_argument('--num_files', type=int, help='Number of files', default=1000)
    parser.add_argument('--roi', type=int, help='ROI flag', default=True)
    parser.add_argument('--output_folder', type=str, help='Output folder')


    args = parser.parse_args() 
    if not args.output_folder:
        args.output_folder = args.folder_path
    folder_path = args.folder_path
    num_files = args.num_files
    roi_flag = args.roi
    output_folder = args.output_folder
    if roi_flag:
        y_width = 4080
        y_height = 764
        y_row_stride = 4080
        chroma_width = 2040
        chroma_height = 382
        chroma_stride = 4080
        chroma_pixel_stride = 2
    else:
        y_width = 4080
        y_height = 3060
        y_row_stride = 4080
        chroma_width = 2040
        chroma_height = 1030
        chroma_stride = 4080
        chroma_pixel_stride = 2

    print(f"Folder path: {folder_path}")
    print(f"Number of files: {num_files}")
    print(f"y_width: {y_width}")
    print(f"y_height: {y_height}") 
    print(f"y_row_stride: {y_row_stride}")
    print(f"chroma_width: {chroma_width}")
    print(f"chroma_height: {chroma_height}")
    print(f"chroma_stride: {chroma_stride}")
    print(f"chroma_pixel_stride: {chroma_pixel_stride}")
    date = extract_date_from_folder_name(os.path.basename(folder_path))
    file_list = []
    for file in os.listdir(folder_path):
        if file.endswith(".rgb"):
            file_path = os.path.join(folder_path, file)
            if file.endswith("config.rgb"):
                config = build_config(file_path, date)   
                continue
            f = YUV_File(file_path, y_width, y_height, y_row_stride, chroma_width, chroma_height, chroma_stride, chroma_pixel_stride)
            frame = f.get_frame()
            time = f.extract_time_from_file_name()
            file_list.append((file_path, frame, time))
            
    file_list.sort(key=lambda x: x[1])
    c = 0
    file_counter = 0
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    movie_file = init_file(os.path.join(output_folder,f"movie_{file_counter}.yuv1"), config)
    for file in file_list:
        f = YUV_File(file[0], y_width, y_height, y_row_stride, chroma_width, chroma_height, chroma_stride, chroma_pixel_stride)
        f.read_file()
        movie_file.write(f.y_data_full)
        movie_file.write(f.u_data)
        movie_file.write(f.v_data)
        movie_file.write(f.extract_time_from_file_name())
        c += 1
        if c == num_files:
            c = 0
            file_counter += 1
            movie_file.close()
            movie_file = init_file(os.path.join(output_folder,f"movie_{file_counter}.yuv1"), config)

            
        
                        

    