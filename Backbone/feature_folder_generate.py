import os


def create_folders_from_txt(txt_path, root_dir):
    # Read video names from the shanghaitech.txt file
    with open(txt_path, 'r') as file:
        video_names = [line.strip() for line in file if line.strip()]

    # Create the folder structure for each video
    for video_name in video_names:
        folder_path = os.path.join(root_dir, "features_video", "i3d", "combine", video_name)
        os.makedirs(folder_path, exist_ok=True)  # Create the folder if it doesn't exist

        # Create an empty feature.npy file in each folder
        feature_file = os.path.join(folder_path, "feature.npy")
        with open(feature_file, 'wb') as f:
            pass  # Create an empty file

    print(f"Successfully created folders for {len(video_names)} videos.")


if __name__ == "__main__":
    txt_path = r"F:\Backbone\video\shanghaitech.txt"  # Path to the shanghaitech.txt file
    root_dir = r"F:\Backbone\model_pkl\10\shanghaitech"  # Root directory for folder creation

    create_folders_from_txt(txt_path, root_dir)
