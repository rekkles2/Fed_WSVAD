import numpy as np
import os
import glob
from multiprocessing import Pool
from tqdm import tqdm
import argparse
import gzip
import pickle
import h5py
def combine_feature():
    import os
    import numpy as np

    # FOLDER PATH
    folder_path = r'F:\Backbone\dataset\shanghaitech\features\i3d\rgb\shanghaitech'
    if os.path.exists(folder_path) == 0:
        os.makedirs(folder_path)
    # Read the file and add a time dimension
    data_list = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            data = np.load(file_path)
            data = np.expand_dims(data, axis=0)
            data_list.append(data)
    stacked_data = np.concatenate(data_list, axis=0)
    output_file = 'F:\\Backbone\\dataset\\shanghaitech\\features_video\\i3d\\combine\\shanghaitech\\feature.npy'
    np.save(output_file, stacked_data)
    print("Saved as an .npy file:", output_file)


def creat_i3d():
    combine_feature()



if __name__=='__main__':
    creat_i3d()

