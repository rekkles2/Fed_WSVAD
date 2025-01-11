import time
from feature_extract import *
from dataset_creater import *
from dataset.write_data_label_txt_new import *
import os
import shutil
from pathlib import Path


# Traverse through the source and destination folders
def copy(source_folder, destination_folder):
    destination_folder = Path(destination_folder)
    if not destination_folder.exists():
        destination_folder.mkdir(parents=True, exist_ok=True)

    try:
        file_list = list(Path(source_folder).iterdir())
    except OSError as e:
        print(f"Error reading directory {source_folder}: {e}")
        return

    npy_files = [f for f in file_list if f.is_file() and f.suffix == '.npy']
    total_files = len(npy_files)

    with tqdm(total=total_files, desc="Moving files") as pbar:
        for file in npy_files:
            source_file = file
            destination_file = destination_folder / file.name
            shutil.move(source_file, destination_file)
            pbar.update(1)


# Delete all files in the specified folder
def delete_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)


def copy_files(source_folder, target_folder):
    file_list = os.listdir(source_folder)
    jpg_files = [filename for filename in file_list if filename.endswith('.jpg')]
    total_files = len(jpg_files)
    with tqdm(total=total_files, desc="Copying files") as pbar:
        for filename in jpg_files:
            source_file = os.path.join(source_folder, filename)
            target_file = os.path.join(target_folder, filename)
            shutil.copy2(source_file, target_file)
            pbar.update(1)


# Example of use

if __name__ == '__main__':
    # Open the file

    """
    shanghaitech.txt should store all video names, for example: 01_001
                                                                01_002
                                                                ......
                                                                13_007
    """
    video_list = "F:\\Backbone\\video\\shanghaitech.txt"
    with open(video_list, "r") as file:
        lines = file.readlines()
        total_videos = len(lines)
        batch_size = 30  # Adjust the batch size according to the actual situation
        num_batches = (total_videos + batch_size - 1) // batch_size

        for i in range(num_batches):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, total_videos)
            batch_lines = lines[start_index:end_index]

            # Process the current batch of videos
            for line in batch_lines:
                # Remove line breaks at the end of lines
                line = line.strip()
                # Take out the file name as the video
                video = line
                frame_folder = f"F:\\Backbone\\video\\Rgb_Fig\\{video}"
                target_folder = r"F:\Backbone\shanghaitech\denseflow\shanghaitech"
                delete_folder = r"F:\Backbone\dataset\shanghaitech\features"
                delete_files_in_folder(target_folder)
                start_time = time.time()
                copy_files(frame_folder, target_folder) # Copy the file
                end_time = time.time()
                execution_time = end_time - start_time
                print("Execution time:", execution_time)
                write_label() # write_data_label_txt_new.py
                # Add an action to delete the destination folder and its internal files before the creat_i3d()
                if os.path.exists(delete_folder):
                    shutil.rmtree(delete_folder)
                i3d_function() # feature_extract.py
                creat_i3d()
                video = str(line)
                ft_folders = 'F:\\Backbone\\dataset\\shanghaitech\\features_video\\i3d\\combine\\shanghaitech'
                feature_folder = f'F:\\Backbone\\model_pkl\\10\\shanghaitech\\features_video\\i3d\\combine\\{video}'
                copy(ft_folders, feature_folder)

                print(f'Video: {video}\nExtraction completed')
                remaining_videos = total_videos - (i + 1) * batch_size
                # Output the number of remaining videos
                print(f"Number of Remaining Videos: {remaining_videos}\n-----------------------------------------------------")
            remaining_videos = total_videos - (i + 1) * batch_size
            print(f"Number of Remaining Videos:{remaining_videos}\n-----------------------------------------------------")
