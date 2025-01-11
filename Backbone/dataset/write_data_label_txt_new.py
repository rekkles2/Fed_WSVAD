import numpy as np
import os
import glob


def main(original_video_dict, dataset, dataset_mode):
    if dataset == 'UCF_Crime':
        zfill_number = 6
    else:
        zfill_number = 5  # ped1, ped2, shanghaitech or avenue

    if os.path.exists('./dataset/{}/{}'.format(dataset, dataset_mode)) == 0:
        os.makedirs('./dataset/{}/{}'.format(dataset, dataset_mode))

    with open(file='./dataset/{}/{}/rgb_list.txt'.format(dataset, dataset_mode), mode='w', encoding='utf-8') as f:
        with open(file='./dataset/{}/{}/label.txt'.format(dataset, dataset_mode), mode='w', encoding='utf-8') as t:
            for k, v in original_video_dict.items():
                # 获取所有img_开头的jpg文件
                frames = glob.glob(os.path.join(v, 'img_*.jpg'))
                frames_number = len(frames)
                framegt = np.zeros(shape=(frames_number), dtype='int8')
                classgt = np.zeros(shape=(frames_number), dtype='int8')
                for i in range(1, frames_number + 1, 1):
                    f.write(os.path.join(v, 'img_' + str(i).zfill(zfill_number) + '.jpg' + '\n'))
                    t.write(str(framegt[i - 1]) + ':' + str(classgt[i - 1]) + '\n')


def write_label():
    data_root = 'F:\Backbone'
    dataset = 'shanghaitech'
    dataset_mode = 'i3d'  # i3d or c3d
    original_video_dict = {}

    if dataset == 'UCF_Crime':
        videopaths = glob.glob(os.path.join(data_root, dataset, 'denseflow', '*/*'))
    else:
        videopaths = glob.glob(os.path.join(data_root, dataset, 'denseflow', '*'))  # ped1, ped2, shanghaitech or avenue
    videonames = [videopath.split('/')[-1] for videopath in videopaths]
    for videopath in videopaths:
        videoname = videopath.split('/')[-1]
        original_video_dict[videoname] = os.path.join(videopath)
    if os.path.exists('./dataset/{}/{}'.format(dataset, dataset_mode)) == 0:
        os.makedirs('./dataset/{}/{}'.format(dataset, dataset_mode))
    np.savetxt('./dataset/{}/{}/videoname.txt'.format(dataset, dataset_mode), np.asarray(videonames).reshape(-1), fmt='%s')
    main(original_video_dict=original_video_dict,
         dataset=dataset,
         dataset_mode=dataset_mode)

if __name__=='__main__':
    write_label()