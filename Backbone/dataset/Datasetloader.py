import sys
sys.path.append("..")
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import argparse

def video_to_tensor(pic):
    """Convert a ``numpy.ndarray`` to tensor.
    Converts a numpy.ndarray (T x H x W x C)
    to a torch.FloatTensor of shape (C x T x H x W)

    Args:
         pic (numpy.ndarray): Video to be converted to tensor.
    Returns:
         Tensor: Converted video.
    """
    return torch.from_numpy(pic.transpose([3, 0, 1, 2]))

class trainDataset(Dataset):#用于加载和预处理视频帧数据，以便在训练深度学习模型时使用。

    def __init__(self, list_file, GT_file,transform=None, cliplen=16, datamodal ='rgb', args=None):
        '''
        Args:
          GT_Dir: (str) path to Ground True dir
          list_file: (str) path to index file.
        '''

        #read list
        with open(list_file) as f:
            self.filelist = f.readlines()
            self.num_samples = len(self.filelist)
        with open(GT_file) as f:
            self.labellist = f.readlines()
        self.transform = transform
        self.cliplen = cliplen
        self.datamodal = datamodal
        self.args = args
        if self.datamodal == 'rgb':
            self.channel = 3
        elif self.datamodal == 'flow' or self.datamodal == 'flownet':
            self.channel = 2
        else:
            raise ('datamodal should be rgb or flow')

    def __getitem__(self, index):
        '''shanghaitech_reconstruct.py

        :param idx: (int) image index
        :return:

        '''
        files = []
        fileinputs = self.filelist[index]
        fileinputs_s = fileinputs.split(' ')
        filedata= []
        # Each file path in the list, reads the corresponding image file, and converts it to RGB format
        for i,fileinput in enumerate(fileinputs_s):
            fileinput = fileinput.replace('\n', '')
            if self.datamodal == 'rgb':
                files.append(fileinput)
                # img = jpeg.JPEG(fileinput).decode()[:, :, [2, 1, 0]]
                img = cv2.imread(fileinput)[:, :, [2, 1, 0]]
                w, h, c = img.shape
                if w < 226 or h < 226:
                    d = 226. - min(w, h)
                    sc = 1 + d / min(w, h)
                    img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
                img = (img / 255.) * 2 - 1
            elif self.datamodal == 'flow' or self.datamodal == 'flownet':
                file_X = fileinput.split('&')[0]
                file_Y = fileinput.split('&')[1]
                files.append(fileinput)
                # flow_x = jpeg.JPEG(file_X).decode()[:, :, 0]
                imgx = cv2.imread((file_X), cv2.IMREAD_GRAYSCALE)
                # flow_y = jpeg.JPEG(file_Y).decode()[:, :, 0]
                imgy = cv2.imread((file_Y), cv2.IMREAD_GRAYSCALE)
                w, h = imgx.shape
                if w < 224 or h < 224:
                    d = 224. - min(w, h)
                    sc = 1 + d / min(w, h)
                    imgx = cv2.resize(imgx, dsize=(0, 0), fx=sc, fy=sc)
                    imgy = cv2.resize(imgy, dsize=(0, 0), fx=sc, fy=sc)

                imgx = (imgx / 255.) * 2 - 1
                imgy = (imgy / 255.) * 2 - 1
                img = np.asarray([imgx, imgy]).transpose([1, 2, 0])
            # h, w, c = img.shape
            # if w < 224 or h < 224:
            #     d = 224. - min(w, h)
            #     sc = 1 + d / min(w, h)
            #     img = cv2.resize(img, dsize=(0, 0), fx=sc, fy=sc)
            img = cv2.resize(img, (224, 224))

            filedata.append(img)

        return video_to_tensor(np.asarray(filedata, dtype=np.float32)), files

    def __len__(self):
        return self.num_samples

# class testDataset(Dataset):
#
#     def __init__(self, list_file, GT_file, transform=None,cliplen=16,
#         mean = './dataset/shanghaitech/unary/normal_mean_frame_227.npy'):
#         '''
#         Args:
#           GT_Dir: (str) path to Ground True dir
#           list_file: (str) path to index file.
#
#         '''
#
#         #read list
#         with open(list_file) as f:
#             self.filelist = f.readlines()
#             self.num_samples = len(self.filelist)
#         with open(GT_file) as f:
#             self.labellist = f.readlines()
#         self.transform = transform
#         self.cliplen = cliplen
#         self.cliplen = cliplen
#         if mean:
#             self.mean = torch.from_numpy(np.load(mean).astype('float32'))
#     def __getitem__(self,index):
#         '''
#
#         :param idx: (int) image index
#         :return:
#
#         '''
#
#         files = []
#         fileinputs = self.filelist[index]
#         labelinputs = self.labellist[index]
#
#         fileinputs_s = fileinputs.split(' ')
#         labelinputs = labelinputs.split(' ')
#         cliplabel, clip_ano_score = self.getcliplabel(labelinputs)
#
#         filedata=torch.empty((self.cliplen, 227, 227))
#         for i, fileinput in enumerate(fileinputs_s):
#             fileinput = fileinput.replace('\n','')
#             files.append(fileinput)
#             # img = cv2.imread(fileinput)[:,:,0]
#             img = jpeg.JPEG(fileinput).decode()[:,:,0]
#             img = img.astype('float32')
#             # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
#             # img = cv2.resize(img, (227, 227), interpolation=cv2.INTER_CUBIC)
#             img = img[np.newaxis, :, :]
#             if self.transform is not None:
#                 img = self.transform(img)
#             else:
#                 img = img / 255.
#                 img = torch.from_numpy(img)
#             if self.mean is not None:
#                 img = img - self.mean
#             filedata[i] = img
#
#         return filedata, torch.from_numpy(clip_ano_score), files
#
#
# # TODO(wanboyang)  all numbers in the matrix cliplabel are same
#     def getcliplabel(self, labels):
#         cliplabel=np.empty(1,np.dtype('int16'))
#         frame_ano_scores=np.empty(len(labels),np.dtype('float32'))
#         for i,label in enumerate(labels):
#             frame_ano_score, framelabel=label.split(':')
#             frame_ano_scores[i]=frame_ano_score
#             cliplabel[0]=framelabel
#
#         return cliplabel, frame_ano_scores
#
#     def __len__(self):
#         return self.num_samples





# It is used to process the list of video frame files and tags, and write the processed result to a new file.
# rgb_list.txt，label.txt It's an old file
# rgb_list_numJoints.txt and trainlabel_numJoints.txtis a new file (i.e. 16 video frames are grouped)
def txttans(origin_filelist,origin_labellist,processed_filelist,processed_labellist,numJoints,model='train',framework='reconstruction'):

    if model == 'train':
        with open(origin_filelist,'r') as l:
            with open(origin_labellist, 'r') as lb:
                lists = l.readlines()
                labellists = lb.readlines()
                with open(processed_filelist, 'w') as t:
                    with open(processed_labellist, 'w') as lbt:
                        temp_file = []
                        temp_lab = []
                        video_name = None
                        framenum = 0
                        for file, label in zip(lists, labellists):
                            file=file.replace('\n','')
                            label=label.replace('\n','')
                            file_split = file.split('\\')
                            # if framework == 'reconstruction' and file_split[-2].find('_a_') != -1:
                            #     continue
                            if video_name:
                                if video_name != file_split[-2]:
                                    video_name = file_split[-2]
                                    if len(temp_file) != numJoints:
                                        temp_file.clear()
                                        temp_lab.clear()
                                        framenum = 0
                                        temp_file.append(file)
                                        temp_lab.append(label)
                                elif int(file_split[-1].split('.')[0].split('_')[-1]) < framenum:
                                    temp_file.clear()
                                    temp_lab.clear()
                                    framenum = 0
                                    temp_file.append(file)
                                    temp_lab.append(label)
                                else:
                                    temp_file.append(file)
                                    temp_lab.append(label)
                            else:
                                video_name = file_split[-2]
                                temp_file.append(file)
                                temp_lab.append(label)
                            if len(temp_file) == numJoints:
                                frame_ano_label, clip_sence_class, clip_ano_label = frame2clip_anolabel(temp_lab)
                                t.write(' '.join(temp_file)+'\n')
                                lbt.write(' '.join(frame_ano_label)+'  ')
                                lbt.write(''.join(clip_sence_class) + '  ')
                                lbt.write(''.join(clip_ano_label) + '\n')
                                framenum = int(temp_file[-1].split('\\')[-1].split('.')[0].split('_')[-1])
                                temp_file.clear()
                                temp_lab.clear()
    else:
        with open(origin_filelist, 'r') as l:
            with open(origin_labellist, 'r') as lb:
                lists = l.readlines()
                labellists = lb.readlines()
                with open(processed_filelist, 'w') as t:
                    with open(processed_labellist, 'w') as lbt:

                        temp_file = []
                        temp_lab = []
                        video_name = None
                        i = 0
                        while i < len(labellists):
                            file = lists[i].replace('\n', '')
                            label = labellists[i].replace('\n', '')
                            file_split = file.split('\\')
                            if video_name:
                                if video_name != file_split[-2]:
                                    video_name = file_split[-2]
                                    if len(temp_file) != numJoints:
                                        temp_file.clear()
                                        temp_lab.clear()
                                        temp_file.append(file)
                                        temp_lab.append(label)
                                        i += 1
                                else:
                                    temp_file.append(file)
                                    temp_lab.append(label)
                                    i += 1
                            else:
                                video_name = file_split[-2]
                                temp_file.append(file)
                                temp_lab.append(label)
                                i += 1
                            if len(temp_file) == numJoints:
                                t.write(' '.join(temp_file) + '\n')
                                lbt.write(' '.join(temp_lab) + '\n')
                                temp_file.clear()
                                temp_lab.clear()
                                i += 1
                                i = i - numJoints



def frame2clip_anolabel(framelabels=None):

    frame_ano_label = []
    clip_sence_class = []
    for i, label in enumerate(framelabels):
        frame_ano_score, framelabel = label.split(':')
        frame_ano_label.append(frame_ano_score)
        clip_sence_class.append(framelabel)

    if np.count_nonzero(np.asarray(frame_ano_label,dtype='int')) >= len(frame_ano_label) - np.count_nonzero(np.asarray(frame_ano_label,dtype='int')):
        clip_ano_label = 1
    else:
        clip_ano_label = 0

    return  frame_ano_label,clip_sence_class[0],str(clip_ano_label)







if __name__ == '__main__':
    datamodal = 'flow'
    origin_filelist = 'E:\\AR\\anomly_feature.pytorch-main\\dataset\\shanghaitech\\i3d\\{}_list.txt'.format(datamodal)
    origin_labellist = 'E:\\AR\\anomly_feature.pytorch-main\\dataset\\shanghaitech\\i3d\\label.txt'
    trainfile_list = 'E:\\AR\\anomly_feature.pytorch-main\\dataset\\{}\\{}\\{}_list_numJoints.txt'.format('shanghaitech','i3d', datamodal)
    trainlabel_list = 'E:\\AR\\anomly_feature.pytorch-main\\dataset\\{}\\{}\\trainlabel_numJoints.txt'.format('shanghaitech','i3d')
    numJoints = 16
    txttans(origin_filelist=origin_filelist,
            origin_labellist=origin_labellist,
            processed_filelist=trainfile_list,
            processed_labellist=trainlabel_list,
            numJoints=numJoints,
            model='train',
            framework=' ')
    # txttans(origin_filelist=origin_testfile_list,
    #         origin_labellist=origin_testlabel_list,
    #         processed_filelist=testfile_list ,
    #         processed_labellist=testlabel_list,
    #         numJoints=numJoints,
    #         model='test')
    trans = transforms.Compose(transforms=[
        transforms.ToTensor()
        # transforms.Normalize(mean=[])
    ])
    train_dataset = trainDataset(list_file=trainfile_list,
                                 GT_file=trainlabel_list, transform=None, cliplen=numJoints, datamodal='flow')
    #
    train_loader = DataLoader(dataset=train_dataset, batch_size=2, pin_memory=True,
                              num_workers=5, shuffle=False)
    #
    for epoch in range(2):
        for i, data in enumerate(train_loader):
            # 将数据从 train_loader 中读出来,一次读取的样本数是32个
            filedata, fileinputs = data
            #
            # # 将这些数据转换成Variable类型
            # inputs = Variable(imagedata)

            # 接下来就是跑模型的环节了，我们这里使用print来代替
            print("epoch：", epoch, "的第", i, "个inputs", filedata.shape, fileinputs)