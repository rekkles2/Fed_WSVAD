# coding:utf-8
import numpy as np
import os
import glob

def main(original_video_dict,dataset,dataset_mode):
    if dataset == 'UCF_Crime':
        zfill_number = 6
    else:
        zfill_number = 5 # ped1, ped2, shanghaitech or avenue
    if dataset_mode == 'i3d':
        if os.path.exists('./dataset/{}/{}'.format(dataset, dataset_mode)) == 0:
            os.makedirs('./dataset/{}/{}'.format(dataset, dataset_mode))
        with open(file='./dataset/{}/{}/rgb_list.txt'.format(dataset, dataset_mode), mode='w', encoding='utf-8') as f:
            with open(file='./dataset/{}/{}/label.txt'.format(dataset, dataset_mode), mode='w', encoding='utf-8') as t:
                for k, v in original_video_dict.items():
                    frames = os.listdir(os.path.join(v))
                    frames_number = int(len(frames))
                    framegt = np.zeros(shape=(frames_number), dtype='int8')
                    classgt = np.zeros(shape=(frames_number), dtype='int8')
                    # Write the path of the video frame and the corresponding label to two different text files, rgb_list.txt and label.txt respectively
                    for i in range(1, frames_number + 1):
                        f.write(os.path.join(v, 'img_' + str(i).zfill(zfill_number) + '.jpg' + '\n'))
                        t.write(str(framegt[i - 1]) + ':' + str(classgt[i - 1]) + '\n')
    else:
        if os.path.exists('./dataset/{}/{}'.format(dataset, dataset_mode)) == 0:
            os.makedirs('./dataset/{}/{}'.format(dataset, dataset_mode))
        with open(file='./dataset/{}/{}/rgb_list.txt'.format(dataset, dataset_mode),mode='w',encoding='utf-8') as f:
                with open(file='./dataset/{}/{}/label.txt'.format(dataset,dataset_mode),mode='w',encoding='utf-8') as t:
                    for k, v in original_video_dict.items():
                        frames = os.listdir(os.path.join(v))
                        frames_number = int(len(frames)/3)
                        framegt = np.zeros(shape=(frames_number), dtype='int8')
                        classgt = np.zeros(shape=(frames_number), dtype='int8')
                        for i in range(1, frames_number + 1, 1):
                            f.write(os.path.join(v, 'img_'+str(i).zfill(zfill_number)+ '.jpg'+'\n'))
                            t.write(str(framegt[i-1])+':'+str(classgt[i-1])+'\n')

def write_label():
    # The data_root and file_path need to be modified
    data_root = r'D:\Backbone'
    file_path = 'D:\Backbone\\dataset\\shanghaitech\\i3d\\rgb_list.txt'
    # No modification is required dataset_mode and dataset
    dataset = 'shanghaitech'
    dataset_mode = 'i3d'  # i3d
    original_video = []
    original_video_dict = {}

    if dataset == 'UCF_Crime':
        videopaths = glob.glob(os.path.join(data_root, dataset, 'denseflow', '*/*'))
    else:
        videopaths = glob.glob(os.path.join(data_root, dataset, 'denseflow', '*')) #ped1, ped2, shanghaitech or avenue
    videonames = []
    for videopath in videopaths:
        videoname = videopath.split('/')[-1]
        videonames.append(videoname)
        original_video_dict[videoname] = os.path.join(videopath)
    if os.path.exists('./dataset/{}/{}'.format(dataset, dataset_mode)) == 0:
        os.makedirs('./dataset/{}/{}'.format(dataset, dataset_mode))
    np.savetxt('./dataset/{}/{}/videoname.txt'.format(dataset, dataset_mode), np.asarray(videonames).reshape(-1),fmt='%s')
    main(original_video_dict=original_video_dict,
         dataset=dataset,
         dataset_mode=dataset_mode)
    # Define the file path

# Read the contents of the file, removing the first two lines
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    lines = lines[2:]
    with open(file_path, 'w', encoding='utf-8') as file:
        file.writelines(lines)

if __name__=='__main__':
    write_label()