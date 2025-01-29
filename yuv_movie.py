import os
import numpy as np
from datetime import datetime, timedelta
import pytz 
import cv2

def downsampleY(y):
    reshapeY = y.reshape((y.shape[0] // 2, 2, y.shape[1] // 2, 2))
    resizeY = reshapeY.mean(axis = (1,3)).round()
    return resizeY.astype('uint8')

def yuv2rgb(y, u, v):
    yuv = np.stack((y,u,v),-1)
    rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)
    return rgb

def decode_timestamp(milliseconds_from_midnight, date):
        time_part = timedelta(milliseconds=milliseconds_from_midnight)  # Convert to timedelta
        
        # Calculate full datetime by adding time part to the base date
        datetime_result = date+ time_part
        Jerusalem_tz = pytz.timezone('Asia/Jerusalem')
        datetime_Jerusalem_tz = datetime_result.astimezone(Jerusalem_tz)
        return datetime_Jerusalem_tz

def calc_color_statistic(img, line, col, win_size):
    start_line = max(line - win_size // 2, 0)
    end_line = min(line + win_size // 2, img.shape[0])

    start_col = max(col - win_size // 2, 0)
    end_col = min(col + win_size // 2, img.shape[1])

    window = img[start_line:end_line, start_col:end_col, :]

    stats = {
        'R_mean': np.mean(window[:, :, 0]),
        'G_mean': np.mean(window[:, :, 1]),
        'B_mean': np.mean(window[:, :, 2]),
        'R_std': np.std(window[:, :, 0]),
        'G_std': np.std(window[:, :, 1]),
        'B_std': np.std(window[:, :, 2])

    }
    return stats



class YUV_Movie:
    def __init__(self, yuv_file_path):
        # Initialize with the YUV file path and predefined sizes
        self.yuv_file = yuv_file_path  # Path to the YUV file
        self.header_size = 1024        # Size of the header in bytes
        self.file_size = os.path.getsize(yuv_file_path)  # Get the file size in bytes
        
        # Dimensions of Y, U, and V components in the YUV file
        self.y_width = 4080
        self.y_height = 764
        self.u_width = 2040
        self.u_height = 382
        self.v_width = self.u_width
        self.v_height = self.u_height

        # Calculate frame size based on YUV dimensions
        self.y_size = self.y_width * self.y_height
        self.u_size = self.u_width * self.u_height
        self.v_size = self.v_width * self.v_height
        self.timestamp_size = 4  # Size of the timestamp in bytes
        self.frame_size = self.y_size + self.u_size + self.v_size + self.timestamp_size  # +4 for timestamp
        num_frames = (self.file_size - self.header_size) / self.frame_size
        if num_frames.is_integer():
            self.num_frames = int(num_frames)
        else:
            raise ValueError("Invalid YUV file: frame size does not match file size")

        self.file = None  # File object for reading the YUV file
        self.Date = None      # Date
        self.ISO = None         # ISO sensitivity
        self.Shutter = None   # Shutter speed
        self.WB = None      # White balance
        self.Focus = None # Foucus
        self.read_header()



    def open_file(self):
        # Open the YUV file in binary read mode
        self.file = open(self.yuv_file, 'rb')

    def read_header(self):
        # Reads and decodes the header from the YUV file
        if not self.file:
            self.open_file()  # Ensure file is open
        self.decode_header(self.file.read(self.header_size))  # Read header bytes

    def decode_header(self, header):
        # Decode the header to extract metadata
        # Extract and parse date from first 4 bytes as uint32
        date_uint32 = np.frombuffer(header[0:4], dtype=np.uint32)[0]
        year = date_uint32 // 10000
        month = (date_uint32 // 100) % 100
        day = date_uint32 % 100
        self.Date = datetime(year, month, day)  # Create datetime object for the date
        
        # Extract other header fields
        self.ISO = np.frombuffer(header[4:6], dtype=np.uint16)[0]          # ISO sensitivity
        self.Shutter = np.frombuffer(header[6:10], dtype=np.float32)[0]    # Shutter speed
        self.WB = np.frombuffer(header[10:12], dtype=np.uint16)[0]         # White balance
        self.Focus = np.frombuffer(header[12:14], dtype=np.uint16)[0]      # Focus level

    def read_frame(self, frame_number):
        if frame_number >= self.num_frames:
            print("Frame number out of range")
            return None
        # Reads a specific frame based on the frame number
        if not self.file:
            self.open_file()  # Ensure file is open
        
        # Calculate frame position and seek to the frame's start position
        self.file.seek(self.header_size + frame_number * self.frame_size)
        
        # Read the frame data and decode it
        frame = self.file.read(self.frame_size)
        return self.decode_frame(frame)
    
    def read_timestamp(self, frame_number):
        if frame_number >= self.num_frames:
            print("Frame number out of range")
            return None
        # Reads a specific frame based on the frame number
        if not self.file:
            self.open_file()  # Ensure file is open

        # Calculate frame position and seek to the frame's start position
        self.file.seek(self.header_size + frame_number * self.frame_size + self.y_size + self.u_size + self.v_size)

        # Read the frame data and decode it
        timestamp = self.file.read(self.timestamp_size)
        milliseconds_from_midnight = int(np.frombuffer(timestamp, dtype=np.uint32)[0])
        datetime_Jerusalem_tz = decode_timestamp(milliseconds_from_midnight, self.Date)
        return datetime_Jerusalem_tz

    def calc_average_fps(self):
        times = [self.read_timestamp(i) for i in range(self.num_frames)]
        time_diffs = [(times [i+1] - times [i]).total_seconds() for i in range(len(times) - 1)]
        average_frame_duration = np.mean(time_diffs)
        average_fps = 1 / average_frame_duration if average_frame_duration > 0 else 0
        return average_fps

    def decode_frame(self, frame):
        # Decode YUV frame and extract the timestamp
        # Extract Y component
        y = np.frombuffer(frame[:self.y_size], dtype=np.uint8).reshape(self.y_height, self.y_width)
        
        # Extract U component
        u = np.frombuffer(frame[self.y_size:self.y_size+self.u_size], dtype=np.uint8).reshape(self.u_height, self.u_width)
        
        # Extract V component
        v = np.frombuffer(frame[self.y_size+self.u_size:self.y_size+self.u_size+self.v_size], dtype=np.uint8).reshape(self.v_height, self.v_width)
        
        # Extract the timestamp in milliseconds from midnight
        milliseconds_from_midnight = int(np.frombuffer(frame[self.y_size+self.u_size+self.v_size:], dtype=np.uint32)[0])
        datetime_Jerusalem_tz = decode_timestamp(milliseconds_from_midnight, self.Date)
                
        return y, u, v, datetime_Jerusalem_tz  # Return decoded frame data and timestamp

    def close_file(self):
        # Close the YUV file if it's open
        if self.file:
            self.file.close()
            self.file = None

    def read_rgb_frame(self, frame):
        y_d, u, v, timestamp = self.read_frame(frame)
        y = downsampleY(y_d)
        rgb = yuv2rgb(y, u, v)
        return rgb

    def disp_frame(self, frame):
        y_d, u, v, timestamp = self.read_frame(frame)
        y = downsampleY(y_d)
        rgb = yuv2rgb(y, u, v)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow('RGB Movie', bgr)

    def play_movie(self, first_frame=0, last_frame=-1):
        # Play the movie by iterating through all frames
        if last_frame == -1:
            last_frame = self.num_frames - 1
        if first_frame < 0 or last_frame >= self.num_frames:
            print("Invalid frame range")
            return

        for i in range(first_frame,last_frame):
            #print(f"Frame {i+1}/{self.num_frames}")
            y_d, u, v, timestamp = self.read_frame(i)
            y = downsampleY(y_d)
            rgb = yuv2rgb(y, u, v)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imshow('RGB Movie', bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                break
        cv2.destroyAllWindows()

    def __str__(self):
        # String representation for the YUV_Movie object
        return f"""YUV_Movie: {self.yuv_file}
Date: {self.Date}
ISO: {self.ISO}
Shutter: {self.Shutter}
White Balance: {self.WB}
Focus: {self.Focus}
Number of Frames: {self.num_frames}
        """
