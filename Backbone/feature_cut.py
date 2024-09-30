import os
import shutil

def copy(source_folders,destination_folders):
    for source_folder, destination_folder in zip(source_folders, destination_folders):
        # Get all the files in the source folder
        files = os.listdir(source_folder)

        for file in files:
            # BUILD SOURCE AND TARGET FILE PATHS
            source_file = os.path.join(source_folder, file)
            destination_file = os.path.join(destination_folder, file)

            # CUT THE FILE TO THE DESTINATION FOLDER
            shutil.move(source_file, destination_file)


if __name__=='__main__':
    video = 'normal_scene_3_scenario_3_1'
    # The source folder path does not need to be changed
    source_folders = [r'E:\AR\anomly_feature.pytorch-main\dataset\shanghaitech\features_video\i3d\combine\shanghaitech']

    print(f'The characteristics of this generation are: {video}')
    # The destination folder path, which is the folder location where you are going to store the extracted features
    destination_folders = [f'D:\\UBnormal-feature\\features_video\\i3d\\combine\\{video}']
    copy(source_folders, destination_folders)
