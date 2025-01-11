import timeit
import os
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import numpy as np
import argparse
from dataset.Datasetloader import trainDataset, txttans
from pytorch_i3d import InceptionI3d
import sys
import winsound
import torchvision.transforms as transforms
from timm.models import create_model
import models


# Load the pretrained model, extract the features of the video frames, and save those features to the specified directory
# ckpt_path is the location of the pretrained model for storing VIDEO_MAE_V2
def feature(data_path, dataset, snapshot, modelName, dataloader, ckpt_path, datamodal='rgb', fc_layer=None):
    """
        Args:
            num_classes (int): Number of classes in the data
            num_epochs (int, optional): Number of epochs to train for.
    """
    ######################build model#####################################
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_params = []
    model = create_model(
        'vit_giant_patch14_224',
        img_size=224,
        pretrained=False,
        num_classes=710,
        all_frames=16,
        tubelet_size=2,
        drop_path_rate=0.3,
        use_mean_pooling=True)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    for model_key in ['model', 'module']:
        if model_key in ckpt:
            ckpt = ckpt[model_key]
            break
    model.load_state_dict(ckpt)
    model.to(device)
    feature_save_dir = os.path.join(data_path, 'dataset', dataset, 'features', modelName, datamodal)

    if os.path.exists(feature_save_dir) == 0:
        os.makedirs(feature_save_dir)
######################log#####################################
    if os.path.exists(os.path.join('./model_feature/', dataset, modelName)) == 0:
        os.makedirs(os.path.join('./model_feature/', dataset, modelName))
    with open(file=os.path.join('./model_feature/', dataset, modelName,'feature.txt'), mode='a+') as f:
        f.write("dataset:{} ".format(dataset)+ '\n')
        f.write("snapshot:{} ".format(snapshot) + '\n')
        f.write("savedir:{} ".format(feature_save_dir) + '\n')
        f.write("========================================== " + '\n')

    model_feature(model=model,dataloader=dataloader, feature_save_dir=feature_save_dir,datamodal=datamodal,dataset=dataset)



def model_feature(model, dataloader, feature_save_dir, datamodal, dataset, feature_layer=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    start_time = timeit.default_timer()
    if dataset=='shanghaitech':
        video_name_po = -2
    else:
        video_name_po = -3
    for img, fileinputs in tqdm(dataloader):
        model.train(False)
        # move inputs and labels to the device the training is taking place on
        inputs = img.to(device)
        # fileinputs = np.asarray(fileinputs)
        fileinputs = np.asarray(fileinputs).transpose((1, 0))
        with torch.no_grad():
            # [10, 2048, 1408] -> [mean, std, max, min]

            features = model.forward_features(inputs)


        features = features.data.cpu().numpy()
        torch.cuda.empty_cache()
        for (fileinput, feature) in zip(fileinputs, features):
            if datamodal == 'flow' or datamodal == 'flownet':
                video_name = fileinput[0].split('&')[0].split('\\')[video_name_po]
                start_frame = fileinput[0].split('&')[0].split('\\')[-1].split('.')[0].split('_')[-1]
                end_frame = fileinput[-1].split('&')[0].split('\\')[-1].split('.')[0].split('_')[-1]
                save_path = os.path.join(feature_save_dir, video_name, start_frame + '_' +end_frame + '.npy')
            else:
                video_name = fileinput[0].split('\\')[video_name_po]
                start_frame = fileinput[0].split('\\')[-1].split('.')[0].split('_')[-1]
                end_frame = fileinput[-1].split('\\')[-1].split('.')[0].split('_')[-1]
                save_path = os.path.join(feature_save_dir, video_name, start_frame + '_' +end_frame + '.npy')

            if os.path.exists(os.path.join(feature_save_dir, video_name)) == 0:
                os.makedirs(os.path.join(feature_save_dir, video_name))
            np.save(save_path, feature)
    stop_time = timeit.default_timer()
    print("Execution time: " + str(stop_time - start_time) + "\n")




def i3d_function():
    # You need to change all the D:\\VAD\Video_MAE\\ in this to the same location in your device, you only need to change this prefix, and the rest of the content does not need to be changed
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device being used :", device)
    a = 'flow'
    b = 'rgb'
    data_path = 'F:\Backbone'
    ckpt_path = r"F:\Backbone\model_pkl\vit_g_hybrid_pt_1200e_k710_ft.pth"
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot1", help='path of testing model_weight', default='D:\Backbone\\shanghaitech\\denseflow\\i3d\\{}_charades.pt'.format(a), type=str)
    parser.add_argument("--snapshot2", help='path of testing model_weight',default='D:\Backbone\\shanghaitech\\denseflow\\i3d\\{}_charades.pt'.format(b), type=str)
    parser.add_argument("--datamodal1", help='rgb or flow', default='flow', type=str)
    parser.add_argument("--datamodal2", help='rgb or flow', default='rgb', type=str)
    parser.add_argument("--dataset", help='Name of dataset', default='shanghaitech', type=str)
    parser.add_argument("--modelName", help='Name of model', default='i3d', type=str)
    parser.add_argument("--fc_layer", help='layer of feature extraction', default='fc6', type=str)
    args = parser.parse_args()
    snapshot1 = args.snapshot1
    snapshot2 = args.snapshot2
    Dataset = args.dataset
    datamodal1 = args.datamodal1
    datamodal2 = args.datamodal2
    origin_filelist2 = 'F:\Backbone\dataset\\shanghaitech\\i3d\\{}_list.txt'.format(datamodal2)
    origin_labellist = 'F:\Backbone\\dataset\\shanghaitech\\i3d\\label.txt'
    trainfile_list2 = '.\\dataset\\{}\\{}\\{}_list_numJoints.txt'.format(Dataset, args.modelName, datamodal2)
    trainlabel_list = '.\\dataset\\{}\\{}\\trainlabel_numJoints.txt'.format(Dataset,args.modelName)

    numJoints = 16

    txttans(origin_filelist=origin_filelist2,
            origin_labellist=origin_labellist,
            processed_filelist=trainfile_list2,
            processed_labellist=trainlabel_list,
            numJoints=numJoints,
            model='train',
            framework=' ')


    train_dataset2 = trainDataset(list_file=trainfile_list2,
                                  GT_file=trainlabel_list,
                                  transform=None,
                                  cliplen=numJoints,
                                  datamodal=datamodal2,
                                  args=args)

    train_dataloader2 = DataLoader(dataset=train_dataset2, batch_size=2, pin_memory=True,
                                   num_workers=8, shuffle=False)

    modelName = args.modelName  # Options: C3D or I3D
    video_MAE = 'video_MAE'
    print(f'THE-CHARACTERISTICS-OF-THIS-GENERATION-ARE:{video_MAE}')
    feature(data_path=data_path,
            dataset=Dataset,
            snapshot=snapshot2,
            modelName=modelName,
            dataloader=train_dataloader2,
            ckpt_path=ckpt_path,
            datamodal=datamodal2,
            fc_layer=args.fc_layer)

if __name__=='__main__':

    i3d_function()
    print('test')
    print('1')