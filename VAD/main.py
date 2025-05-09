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


    if not os.path.exists('./ckpt/' + save_path):
        os.makedirs('./ckpt/' + save_path)
    if not os.path.exists('./logs/' + save_path):
        os.makedirs('./logs/' + save_path)
    train(epochs=args.max_epoch, train_loader=train_loader, all_test_loader=all_test_loader, args=args, model=model, optimizer=optimizer, device=device, save_path=save_path)





