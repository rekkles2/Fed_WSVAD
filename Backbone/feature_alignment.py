import os
import numpy as np
import pickle


feature_dir = r"\VAD\shanghaitech\features_video\i3d\combine"
gt_path = r"\VAD\shanghaitech\GT\frame_label.pickle"


with open(gt_path, 'rb') as f:
    gt_dict = pickle.load(f)


for video_name in os.listdir(feature_dir):
    video_path = os.path.join(feature_dir, video_name)
    if not os.path.isdir(video_path):
        continue


    npy_files = [f for f in os.listdir(video_path) if f.endswith('.npy')]
    if not npy_files:
        print(f"No npy file found in {video_path}")
        continue

    npy_file = npy_files[0]
    npy_path = os.path.join(video_path, npy_file)


    features = np.load(npy_path)
    T, _, _ = features.shape  # features shape: [T, 3, 1408]


    if video_name not in gt_dict:
        print(f"Warning: No ground truth for video {video_name}")
        continue
    scores = gt_dict[video_name]
    S = len(scores)


    T_expanded = T * 16


    if T_expanded < S:
        scores = scores[:T_expanded]
    else:

        target_T = S // 16
        if T >= target_T:

            features = features[:target_T]
        else:

            features = np.repeat(features, target_T // T + 1, axis=0)[:target_T]


    np.save(npy_path, features)