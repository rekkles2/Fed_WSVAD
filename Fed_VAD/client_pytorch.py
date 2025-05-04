# "Dual-detector Re-optimization for Federated Weakly Supervised Video Anomaly Detection
# via Adaptive Dynamic Recursive Mapping", accepted by IEEE Transactions on Industrial Informatics (TII).
#
# Repository: https://github.com/rekkles2/Fed_WSVAD  
# Maintained by: Jiahang Li  
# License: Apache License 2.0 (see https://github.com/rekkles2/Fed_WSVAD/blob/main/LICENSE)


from sklearn.metrics import roc_auc_score, confusion_matrix
import argparse
import warnings
from collections import OrderedDict
from model import model_generater
import flwr as fl
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
from test import test
import numpy as np
import pickle
from losses import KMXMILL_individual, normal_smooth
from eval import eval_p
import losses
from video_dataset_anomaly_balance_uni_sample import dataset1, dataset2, dataset3, dataset4  # For anomaly
import torch.optim as optim
import datetime
import time
import copy
from SALA import SALA


parser = argparse.ArgumentParser(description="Flower Embedded devices")
parser.add_argument("--server_address", type=str, default="191.162.0.104:8080",help=f"gRPC server address (deafault '192.168.31.238:8080')",)
parser.add_argument("--cid",type=int, required=True, help="Client id. Should be an integer between 0 and NUM_CLIENTS",)
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate (default: 0.0001)')
parser.add_argument('--model_name', default='model_single', help='')
parser.add_argument('--loss_type', default='DMIL_C', type=str,  help='the type of n_pair loss, max_min_2, max_min, attention, attention_median, attention_H_L or max')
parser.add_argument('--pretrain', type=int, default=0)
parser.add_argument('--pretrained_ckpt', default=None, help='ckpt for pretrained model')
parser.add_argument('--testing_path', type=str, default=None, help='time file for test model')
parser.add_argument('--testing_model', type=str, default=None, help='iteration name for testing model')
parser.add_argument('--feature_size', type=int, default=1408, help='size of feature (default: 2048)')
parser.add_argument('--batch_size',  type=int, default=1, help='number of samples in one itration')
parser.add_argument('--sample_size',  type=int, default=2, help='number of samples in one itration')
parser.add_argument('--sample_step', type=int, default=1, help='')
parser.add_argument('--dataset_name', type=str, default='shanghaitech', help='')
parser.add_argument('--dataset_path', type=str, default='/root/Fedmae', help='path to dir contains anomaly datasets')
parser.add_argument('--feature_modal', type=str, default='combine', help='features from different input, options contain rgb, flow , combine')
parser.add_argument('--max-seqlen', type=int, default=300, help='maximum sequence length during training (default: 750)')
parser.add_argument('--Lambda', type=str, default='1_20', help='')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument('--max_epoch', type=int, default=25, help='maximum iteration to train (default: 50000)')
parser.add_argument('--feature_pretrain_model', type=str, default='i3d', help='type of feature to be used I3D or C3D (default: I3D)')
parser.add_argument('--feature_layer', type=str, default='fc6', help='fc6 or fc7')
parser.add_argument('--k', type=int, default=6, help='value of k')
parser.add_argument('--plot', type=int, default=1, help='whether plot the video anomalous map on testing')
# parser.add_argument('--rank', type=int, default=0, help='')
# parser.add_argument('--loss_instance_type', type=str, default='weight', help='mean, weight, weight_center or individual')
# parser.add_argument('--MIL_loss_type', type=str, default='CE', help='CE or MSE')
parser.add_argument('--larger_mem', type=int, default=0, help='')
# parser.add_argument('--u_ratio', type=int, default=10, help='')
# parser.add_argument('--anomaly_smooth', type=int, default=1,
#                     help='type of smooth function, all or normal')
# parser.add_argument('--sparise_term', type=int, default=1,
#                     help='type of smooth function, all or normal')
# parser.add_argument('--attention_type', type=str, default='softmax',
#                     help='type of normalization of attention vector, softmax or sigmoid')
# parser.add_argument('--confidence', type=float, default=0, help='anomaly sample threshold')
parser.add_argument('--snapshot', type=int, default=100, help='anomaly sample threshold')
# parser.add_argument('--ps', type=str, default='normal_loss_mean')
parser.add_argument('--s', type=int, default=20, help='More fine-grained than its original paper.')
parser.add_argument('--p', type=int, default=100, help="rand_percent")
parser.add_argument('--e', type=float, default=0.1, help="eta")
parser.add_argument('--b', type=float, default=0.1, help="beta")
parser.add_argument('--label_type', type=str, default='unary')
parser.add_argument('--t', type=str, default=0.1, help="Train the weight until the standard deviation of the recorded "
                                            "losses is less than a given threshold. Default: 0.1")

# The maximum number of clients that can be connected
warnings.filterwarnings("ignore", category=UserWarning)
NUM_CLIENTS = 50

# Training function
def train(epochs, train_loader, args, model, optimizer, device, save_path):
    global itr
    if os.path.exists(os.path.join('./result', save_path)) == 0:
        os.makedirs(os.path.join('./result', save_path))
    with open(file=os.path.join('./result', save_path, 'result.txt'), mode='w') as f:
        for key, value in vars(args).items():
            f.write('%s:%s\n' % (key, value))
    log_statics = {}
    if args.pretrained_ckpt:
        checkpoint = torch.load(args.pretrained_ckpt)
        model.load_state_dict(checkpoint)
        print('model load weights from {}'.format(args.pretrained_ckpt))
    else:
        print('model is trained from scratch')
    for epoch in range(epochs):
        for i, data in enumerate(train_loader):
            itr += 1
            [anomaly_features, normaly_features], [anomaly_label, normaly_label], stastics_data = data
            features = torch.cat((anomaly_features.squeeze(0), normaly_features.squeeze(0)), dim=0)
            seq_feature = features[:, :, 0, :]
            videolabels = torch.cat((anomaly_label.squeeze(0), normaly_label.squeeze(0)), dim=0)
            seq_len = torch.sum(torch.max(seq_feature.abs(), dim=2)[0] > 0, dim=1).numpy()
            features = features[:, :np.max(seq_len), :, :]

            # features = torch.from_numpy(features).float().to(device)
            features = features.float().to(device)
            videolabels = videolabels.float().to(device)
            # final_features, element_logits_linear, element_logits = model(features)
            # [B, T, 3, 1408] ->
            mean_score, channel_score, element_logits, R, = model(features)
            # if args.model_name == 'model_lstm':
            #     final_features, element_logits = model(features, seq_len)
            # else:
            #     final_features, element_logits = model(features)
            weights = args.Lambda.split('_')
            m_loss = KMXMILL_individual(element_logits=element_logits,
                                        seq_len=seq_len,
                                        labels=videolabels,
                                        device=device,
                                        loss_type='CE',
                                        args=args)
            m_loss_mean = KMXMILL_individual(element_logits=mean_score,
                                             seq_len=seq_len,
                                             labels=videolabels,
                                             device=device,
                                             loss_type='CE',
                                             args=args)
            m_loss_channel = KMXMILL_individual(element_logits=channel_score,
                                                seq_len=seq_len,
                                                labels=videolabels,
                                                device=device,
                                                loss_type='CE',
                                                args=args)
            n_loss = normal_smooth(element_logits=element_logits,
                                   labels=videolabels,
                                   device=device)
            n_loss_channel = normal_smooth(element_logits=channel_score,
                                           labels=videolabels,
                                           device=device)

            n_loss_mean = normal_smooth(element_logits=mean_score,
                                        labels=videolabels,
                                        device=device)
            L_TV = losses.L_TV().cuda()
            Loss_TV = 300 * L_TV(R)

            total_loss = float(weights[0]) * m_loss + Loss_TV + float(weights[0]) * m_loss_mean + float(weights[0]) * m_loss_channel + \
                    float(weights[1]) * n_loss_mean + float(weights[1]) * n_loss_channel + float(weights[1]) * n_loss

            if itr % 20 == 0 and not itr == 0:
                # print(final_features.shape)
                print('Iteration:{}, Loss: {}'
                      .format(itr, total_loss.data.cpu().detach().numpy()))
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()


def test1(model, save_path, all_test_loader, device, args):
    global itr
    test_loader = all_test_loader
    torch.save(model.state_dict(), os.path.join('./ckpt/', save_path, 'iter_{}'.format(itr) + '.pkl'))
    start_time = time.time()
    test_result_dict = test(test_loader, model, device, args)
    end_time = time.time()
    print(start_time - end_time)
    # train_result_dict = test(train2test_loader, model, device, args)
    abnormal_auc_score, all_auc_score = eval_p(itr=itr, dataset=args.dataset_name, predict_dict=test_result_dict,
                        save_path=save_path, plot=args.plot, args=args)
    return abnormal_auc_score, all_auc_score

# Partition the dataset
def prepare_dataset(args):
    """Get shanghaitech and return client partitions and global testset."""
    train_dataset1 = dataset1(args=args, train=True)
    train_dataset2 = dataset2(args=args, train=True)
    test_dataset1 = dataset1(args=args, train=False)
    test_dataset2 = dataset2(args=args, train=False)
    train_dataset3 = dataset3(args=args, train=True)
    train_dataset4 = dataset4(args=args, train=True)
    test_dataset3 = dataset3(args=args, train=False)
    test_dataset4 = dataset4(args=args, train=False)
    print("Partitioning dataset (IID)...")
    # Split trainset into `num_partitions` trainsets
    train_partitions = [train_dataset1, train_dataset2, train_dataset3, train_dataset4]
    val_partitions = [test_dataset1, test_dataset2, test_dataset3, test_dataset4]

    return train_partitions, val_partitions


# Flower client, adapted from Pytorch quickstart/simulation example
class FlowerClient(fl.client.NumPyClient):

    def __init__(self, trainset, valset, args, save_path):
        self.trainset = trainset
        self.valset = valset
        # Instantiate model
        self.model = model_generater(model_name=args.model_name, feature_size=args.feature_size)
        self.local_model = model_generater(model_name=args.model_name, feature_size=args.feature_size)
        self.global_model = model_generater(model_name=args.model_name, feature_size=args.feature_size)
        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.local_model.to(self.device)
        self.global_model.to(self.device)# send model to device
        self.args = args
        self.loss = torch.nn.CrossEntropyLoss()
        self.save_path = save_path
        self.SALA = SALA(cid=self.args.cid, train_data=self.trainset, batch_size=1,rand_percent=self.args.p,
                    layer_idx=self.args.s, eta=self.args.e, beta=self.args.b, args=self.args, threshold=self.args.t)

    def local_initialization(self,):
        self.SALA.adaptive_local_aggregation(self.global_model, self.local_model)

    def set_parameters(self, params):                    # Receive model parameters
        """Set model weights from a list of NumPy ndarrays."""
        params_dict = zip(self.model.state_dict().keys(), params)
        state_dict = OrderedDict(
            {
                k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
                for k, v in params_dict
            }
        )
        self.model.load_state_dict(state_dict, strict=True)

    def get_parameters(self, config):                    # Send model parameters
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        print("Client sampled for fit()")
        # Receive a globally initialized model to ensure that the model of each client is the same, and start receiving the local model in the second round
        self.set_parameters(parameters)
        # Read hyperparameters from config set by the server
        batch, epochs = config["batch_size"], config["epochs"]
        # Construct dataloader
        trainloader = DataLoader(dataset=self.trainset, batch_size=batch, pin_memory=True,
                              num_workers=4, shuffle=True)
        # Updated with the latest fine-tuned local model
        self.model = copy.deepcopy(self.local_model)
        # Define optimizer
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=0.0005)
        # Train
        train(epochs=epochs, train_loader=trainloader, args=self.args, model=self.model, optimizer=optimizer, device=self.device, save_path=self.save_path)
        self.local_model = copy.deepcopy(self.model)
        # Return local model and statistics
        return self.get_parameters({}), len(trainloader.dataset), {}

    def evaluate(self, parameters, config):
        print("Client sampled for evaluate()")
        # Receive updated local model parameters
        self.set_parameters(parameters)
        # Update it to the global model
        self.global_model = copy.deepcopy(self.model)
        # Construct dataloader
        valloader = DataLoader(dataset=self.valset, batch_size=1, pin_memory=True,
                             num_workers=1, shuffle=False)
        # Local adaptive aggregation
        self.local_initialization()
        self.model = copy.deepcopy(self.local_model)
        # Evaluate
        abnormal_auc_score, all_auc_score = test1(model=self.model,save_path=self.save_path,all_test_loader=valloader,device=self.device,args=self.args)
        # Return statistics
        return float(all_auc_score), len(valloader.dataset), {"all_auc_score": float(abnormal_auc_score)}


def main():
    global itr
    args = parser.parse_args()
    print(args)
    assert args.cid < NUM_CLIENTS

    time = datetime.datetime.now()
    save_path = os.path.join(args.model_name, args.feature_pretrain_model, args.dataset_name, 'k_{}'.format(args.k),
                             '_Lambda_{}'.format(args.Lambda), args.feature_modal,
                             '{}{:02d}{:02d}{:02d}{:02d}{:02d}'.format(time.year, time.month, time.day, time.hour,time.minute, time.second))
    if not os.path.exists('./ckpt/' + save_path):
        os.makedirs('./ckpt/' + save_path)
    if not os.path.exists('./logs/' + save_path):
        os.makedirs('./logs/' + save_path)
    # Download shanghaitech dataset and partition it
    trainsets, valsets = prepare_dataset(args=args)

    # Start Flower client setting its associated data partition
    fl.client.start_client(
        server_address=args.server_address,
        client=FlowerClient(
            trainset=trainsets[args.cid], valset=valsets[args.cid], args=args, save_path=save_path
        ).to_client(),
        grpc_max_message_length=736870912,
    )


if __name__ == "__main__":
    itr = 0
    main()
