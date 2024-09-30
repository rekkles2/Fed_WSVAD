from __future__ import print_function
import os
import torch
from model import model_generater
from video_dataset_anomaly_balance_uni_sample import dataset, dataset_train2test  # For anomaly
from torch.utils.data import DataLoader
from train import train
import options
import datetime
import glob
import torch.optim as optim
import pickle
import os
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
import sys
from utils import scorebinary, anomap

if __name__ == '__main__':
    args = options.parser.parse_args()
    torch.manual_seed(args.seed)
    device = torch.device("cuda:{}".format(args.device))
    torch.cuda.set_device(args.device)
    time = datetime.datetime.now()

    save_path = os.path.join(args.model_name, args.feature_pretrain_model, args.dataset_name, 'k_{}'.format(args.k), '_Lambda_{}'.format(args.Lambda), args.feature_modal, '{}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(time.year, time.month, time.day, time.hour,time.minute, time.second))

    model = model_generater(model_name=args.model_name, feature_size=args.feature_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0005)
    if args.pretrained_ckpt is not None:
        checkpoint = torch.load(args.pretrained_ckpt)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model.load_state_dict(pretrained_dict, strict=False)

    train_dataset = dataset(args=args, train=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, pin_memory=True,
                                num_workers=1, shuffle=True)
    test_dataset = dataset(args=args, train=False)
    train2test_dataset = dataset_train2test(args=args)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, pin_memory=True,
                                num_workers=0, shuffle=False)
    train2test_loader = DataLoader(dataset=train2test_dataset, batch_size=1, pin_memory=True,
                                num_workers=0, shuffle=False)
    all_test_loader = [train2test_loader, test_loader]

    checkpoint = torch.load(args.inference)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
    model.load_state_dict(pretrained_dict, strict=False)

    result = {}
    for i, data in enumerate(test_loader):
        feature, data_video_name = data
        feature = feature.to(device)
        with torch.no_grad():
            if args.model_name == 'model_lstm':
                _, element_logits = model(feature, seq_len=None, is_training=False)
            else:
                _, channel_score, element_logits, r, = model(feature, is_training=False)
        element_logits = element_logits.cpu().data.numpy().reshape(-1)
        # element_logits = F.softmax(element_logits, dim=2)[:, :, 1].cpu().data.numpy()
        # element_logits = element_logits.cpu().data.numpy()
        result[data_video_name[0]] = element_logits

    dataset = args.dataset_name
    if dataset == 'shanghaitech':
        label_dict_path = '{}/shanghaitech/GT'.format(args.dataset_path)
    elif dataset == 'UBnormal':
        label_dict_path = '{}/UBnormal/GT'.format(args.dataset_path)
    elif dataset == 'avenue':
        label_dict_path = '{}/avenue/GT'.format(args.dataset_path)
    elif dataset == 'ped2':
        label_dict_path = '{}/ped2/GT'.format(args.dataset_path)
    with open(file=os.path.join(label_dict_path, 'frame_label.pickle'), mode='rb') as f:
        frame_label_dict = pickle.load(f)
    with open(file=os.path.join(label_dict_path, 'video_label.pickle'), mode='rb') as f:
        video_label_dict = pickle.load(f)
    all_predict_np = np.zeros(0)
    all_label_np = np.zeros(0)
    normal_predict_np = np.zeros(0)
    normal_label_np = np.zeros(0)
    abnormal_predict_np = np.zeros(0)
    abnormal_label_np = np.zeros(0)
    for k, v in result.items():
        if video_label_dict[k] == [1.]:
            frame_labels = frame_label_dict[k]
            all_predict_np = np.concatenate((all_predict_np, v.repeat(16)))
            all_label_np = np.concatenate((all_label_np, frame_labels[:len(v.repeat(16))]))
            abnormal_predict_np = np.concatenate((abnormal_predict_np, v.repeat(16)))
            abnormal_label_np = np.concatenate((abnormal_label_np, frame_labels[:len(v.repeat(16))]))
        elif video_label_dict[k] == [0.]:
            frame_labels = frame_label_dict[k]
            all_predict_np = np.concatenate((all_predict_np, v.repeat(16)))[:140496]
            all_label_np = np.concatenate((all_label_np, frame_labels[:len(v.repeat(16))]))
            normal_predict_np = np.concatenate((normal_predict_np, v.repeat(16)))
            normal_label_np = np.concatenate((normal_label_np, frame_labels[:len(v.repeat(16))]))

    all_auc_score = roc_auc_score(y_true=all_label_np, y_score=all_predict_np)
    binary_all_predict_np = scorebinary(all_predict_np, threshold=0.5)
    tn, fp, fn, tp = confusion_matrix(y_true=all_label_np, y_pred=binary_all_predict_np).ravel()
    all_ano_false_alarm = fp / (fp + tn)
    binary_normal_predict_np = scorebinary(normal_predict_np, threshold=0.5)
    # tn, fp, fn, tp = confusion_matrix(y_true=normal_label_np, y_pred=binary_normal_predict_np).ravel()
    fp_n = binary_normal_predict_np.sum()
    normal_count = normal_label_np.shape[0]
    normal_ano_false_alarm = fp_n / normal_count

    abnormal_auc_score = roc_auc_score(y_true=abnormal_label_np, y_score=abnormal_predict_np)
    binary_abnormal_predict_np = scorebinary(abnormal_predict_np, threshold=0.5)
    tn, fp, fn, tp = confusion_matrix(y_true=abnormal_label_np, y_pred=binary_abnormal_predict_np).ravel()
    abnormal_ano_false_alarm = fp / (fp + tn)

    print('AUC_score_all_video is {}'.format(all_auc_score))
    print('AUC_score_abnormal_video is {}'.format(abnormal_auc_score))
    print('ano_false_alarm_all_video is {}'.format(all_ano_false_alarm))
    print('ano_false_alarm_normal_video is {}'.format(normal_ano_false_alarm))
    print('ano_false_alarm_abnormal_video is {}'.format(abnormal_ano_false_alarm))


