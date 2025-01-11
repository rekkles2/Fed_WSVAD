import os
import cv2


def extract_frames(video_path, output_folder):
    """
    Extracts frames from a video and saves them as RGB images.

    Args:
        video_path (str): Path to the input video file.
        output_folder (str): Path to the folder where extracted frames will be saved.
    """
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    frame_count = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Exit the loop when the video ends

        # Convert the frame from BGR to RGB format
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Generate the output filename for the frame
        frame_filename = os.path.join(output_folder, f"img_{frame_count:05d}.jpg")

        # Save the frame as a JPEG image
        cv2.imwrite(frame_filename, frame_rgb)
        frame_count += 1

    cap.release()
    print(f"Extracted {frame_count} frames from {video_path}")


def process_folder(root_folder, output_root):
    """
    Recursively processes all .avi videos in the root folder, extracting frames.

    Args:
        root_folder (str): Path to the folder containing video files.
        output_root (str): Path to the folder where extracted frames will be saved.
    """
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith('.avi'):  # Process only .avi files
                video_path = os.path.join(root, file)
                video_name = os.path.splitext(file)[0]  # Get video name without extension

                # Create an output folder in Rgb_Fig with the video name
                output_folder = os.path.join(output_root, video_name)

                if not os.path.exists(output_folder):
                    os.makedirs(output_folder)  # Create the folder if it doesn't exist

                extract_frames(video_path, output_folder)


if __name__ == "__main__":
    # Root folder containing the dataset videos
    root_folder = r"F:\Backbone\video\shanghaitech"

    # Folder where extracted RGB frames will be saved
    output_root = r"F:\Backbone\video\Rgb_Fig"

    # Start processing the videos
    process_folder(root_folder, output_root)
