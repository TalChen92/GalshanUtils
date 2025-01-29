import shutil
import os

# Original file and output folder
original_file = 'yuv_93_07_01_04_118.rgb'
output_folder = 'data_folder'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)
file_name = os.path.basename(original_file)
date_str = file_name.split('_', 1)[1].split('.')[0]
# Parse the initial values from the original filename
frame_number, hour, minute, second, millisecond = map(int, date_str.split('_'))

# Define the number of files to create and the millisecond increment
num_files = 200
millisecond_increment = 30

# Loop to create 2000 files with incremented frame numbers and timestamps
for i in range(num_files):
    # Update frame number
    new_frame_number = frame_number + i

    # Calculate new timestamp by adding 30 milliseconds
    total_milliseconds = millisecond + (i * millisecond_increment)
    new_millisecond = total_milliseconds % 1000
    total_seconds = second + (total_milliseconds // 1000)
    new_second = total_seconds % 60
    total_minutes = minute + (total_seconds // 60)
    new_minute = total_minutes % 60
    new_hour = hour + (total_minutes // 60)

    # Construct the new filename
    new_filename = f'yuv_{new_frame_number}_{new_hour:02}_{new_minute:02}_{new_second:02}_{new_millisecond:03}.rgb'
    new_filepath = os.path.join(output_folder, new_filename)

    # Copy the original file to the new filename
    shutil.copy(original_file, new_filepath)

print(f"Created {num_files} files in the '{output_folder}' directory.")
