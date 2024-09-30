import numpy as np
import torch
import torch.nn as nn
import copy
import random
from torch.utils.data import DataLoader, Subset
from typing import List, Tuple
from losses import KMXMILL_individual, normal_smooth, L_TV
import argparse

class SALA:
    def __init__(self,
                cid: int,
                train_data: List[Tuple],
                batch_size: int, 
                rand_percent: int,
                args: argparse.Namespace,
                layer_idx: int = 0,
                eta: float = 1.0,
                beta: float = 1.0,
                los: float = 2.0,
                device: str = 'cuda:0',
                threshold: float = 0.01,
                num_pre_loss: int = 5,) -> None:
        """
        Initialize SALA module

        Args:
            cid: Client ID.
            train_data: The reference of the local training data.
            batch_size: Weight learning batch size.
            rand_percent: The percent of the local training data to sample.
            layer_idx: Control the weight range. By default, all the layers are selected. Default: 0
            eta: Weight learning rate. Default: 1.0
            beta: Weight learning rate. Default: 1.0
            device: Using cuda or cpu. Default: 'cpu'
            threshold: Train the weight until the standard deviation of the recorded losses is less than a given threshold. Default: 0.1
            num_pre_loss: The number of the recorded losses to be considered to calculate the standard deviation. Default: 5

        Returns:
            None.
        """

        self.cid = cid
        self.train_data = train_data
        self.batch_size = batch_size
        self.rand_percent = rand_percent
        self.layer_idx = layer_idx
        self.eta = eta
        self.beta = beta
        self.threshold = threshold
        self.num_pre_loss = num_pre_loss
        self.device = device
        self.args = args
        self.los = los
        self.weights = None # Learnable local aggregation weights.
        self.alpha = None
        self.start_phase = True


    def adaptive_local_aggregation(self, 
                            global_model: nn.Module,
                            local_model: nn.Module) -> None:
        """
        Generates the Dataloader for the randomly sampled local training data and 
        preserves the lower layers of the update. 

        Args:
            global_model: The received global/aggregated model. 
            local_model: The trained local model. 

        Returns:
            None.
        """

        # randomly sample partial local training data
        rand_ratio = self.rand_percent / 100
        rand_num = int(rand_ratio*len(self.train_data))
        rand_idx = random.randint(0, len(self.train_data)-rand_num)
        indices = list(range(rand_idx, rand_idx + rand_num))
        subset_train_dataset = Subset(self.train_data, indices)
        rand_loader = DataLoader(subset_train_dataset, pin_memory=True, batch_size=1,
                                num_workers=1, shuffle=True)


        # obtain the references of the parameters
        params_g = list(global_model.parameters())
        params = list(local_model.parameters())

        print('Client:', self.cid, 'SALA: Train')

        # preserve all the updates in the lower layers
        for param, param_g in zip(params[:-self.layer_idx], params_g[:-self.layer_idx]):
            param.data = param_g.data.clone()

        # temp local model only for weight learning
        model_t = copy.deepcopy(local_model)
        params_t = list(model_t.parameters())

        # only consider higher layers
        params_p = params[-self.layer_idx:]  # local_model
        params_gp = params_g[-self.layer_idx:]  # global_model
        params_tp = params_t[-self.layer_idx:]

        # frozen the lower layers to reduce computational cost in Pytorch
        for param in params_t[:-self.layer_idx]:
            param.requires_grad = False

        # used to obtain the gradient of higher layers
        # no need to use optimizer.step(), so lr=0
        optimizer = torch.optim.SGD(params_tp, lr=0)

        # initialize the weight to all ones in the beginning
        if self.weights == None:
            self.weights = [torch.ones_like(param.data).to(self.device) for param in params_tp]
        if self.alpha == None:
            self.alpha = [torch.ones_like(param.data).to(self.device) for param in params_tp]

        # initialize the higher layers in the temp local model
        for param_t, param, param_g, weight, alpha in zip(params_tp, params_p, params_gp,
                                                        self.weights, self.alpha):
            param_t.data = alpha * (param + (param_g - param) * weight)

        # weight learning
        losses_weight = []  # record losses
        it = 0
        while True:
            for i, data in enumerate(rand_loader):
                it += 1
                optimizer.zero_grad()
                [anomaly_features, normaly_features], [anomaly_label, normaly_label], stastics_data = data
                features = torch.cat((anomaly_features.squeeze(0), normaly_features.squeeze(0)), dim=0)
                seq_feature = features[:, :, 0, :]
                videolabels = torch.cat((anomaly_label.squeeze(0), normaly_label.squeeze(0)), dim=0)
                seq_len = torch.sum(torch.max(seq_feature.abs(), dim=2)[0] > 0, dim=1).numpy()
                features = features[:, :np.max(seq_len), :, :]

                # features = torch.from_numpy(features).float().to(device)
                features = features.float().to(self.device)
                videolabels = videolabels.float().to(self.device)
                # final_features, element_logits_linear, element_logits = model(features)
                # [B, T, 3, 1408] ->
                mean_score, channel_score, element_logits, R, = model_t(features)
                # if args.model_name == 'model_lstm':
                #     final_features, element_logits = model(features, seq_len)
                # else:
                #     final_features, element_logits = model(features)
                weights = self.args.Lambda.split('_')
                m_loss = KMXMILL_individual(element_logits=element_logits,
                                            seq_len=seq_len,
                                            labels=videolabels,
                                            device=self.device,
                                            loss_type='CE',
                                            args=self.args)
                m_loss_mean = KMXMILL_individual(element_logits=mean_score,
                                                seq_len=seq_len,
                                                labels=videolabels,
                                                device=self.device,
                                                loss_type='CE',
                                                args=self.args)
                m_loss_channel = KMXMILL_individual(element_logits=channel_score,
                                                    seq_len=seq_len,
                                                    labels=videolabels,
                                                    device=self.device,
                                                    loss_type='CE',
                                                    args=self.args)
                n_loss = normal_smooth(element_logits=element_logits,
                                    labels=videolabels,
                                    device=self.device)
                n_loss_channel = normal_smooth(element_logits=channel_score,
                                            labels=videolabels,
                                            device=self.device)

                n_loss_mean = normal_smooth(element_logits=mean_score,
                                            labels=videolabels,
                                            device=self.device)
                L_TV1 = L_TV().cuda()
                Loss_TV = 300 * L_TV1(R)

                loss_value = float(weights[0]) * m_loss + Loss_TV + float(weights[0]) * m_loss_mean + float(weights[0]) * m_loss_channel + \
                            float(weights[1]) * n_loss_mean + float(weights[1]) * n_loss_channel + float(weights[1]) * n_loss
                if it % 20 == 0 and not it == 0:
                    # print(final_features.shape)
                    print('Iteration:{}, Loss: {}'.format(it, loss_value.data.cpu().detach().numpy()))
                    losses_weight.append(loss_value.item())     # A LOSS VALUE IS ADDED EVERY 20 ITERATIONS

                loss_value.backward()

                original_weights = [weight.clone() for weight in self.weights]
                original_alpha = [alpha.clone() for alpha in self.alpha]

                # update weights
                for param_t, param, param_g, weight, alpha in zip(params_tp, params_p, params_gp, self.weights,
                        original_alpha):
                    if param_t.grad is None:
                        continue
                    weight.data = torch.clamp(
                        weight - self.eta * (param_t.grad * alpha * (param_g - param)), 0, 1)

                for param_t, param, param_g, weight, alpha in zip(params_tp, params_p, params_gp
                        , original_weights, self.alpha):
                    if param_t.grad is None:
                        continue
                    alpha.data = torch.clamp(
                        alpha - self.beta * (param_t.grad * (param + weight * (param_g - param))), 0, 2)

                for param_t, param, param_g, weight, alpha in zip(params_tp, params_p, params_gp
                        , self.weights, self.alpha):
                    param_t.data = alpha * (param + (param_g - param) * weight)

            # only train one epoch in the subsequent iterations
            if not self.start_phase:
                break
                # train the weight until convergence
            if len(losses_weight) > self.num_pre_loss and np.std(losses_weight[-self.num_pre_loss:]) < self.threshold and losses_weight[-1] < self.los:
                print('Client:', self.cid, '\tStd:', np.std(losses_weight[-self.num_pre_loss:]),
                        '\tSALA epochs:', it)
                break
            if it >= 5000:
                print('Client:', self.cid, '\tStd:', np.std(losses_weight[-self.num_pre_loss:]),
                        '\tSALA epochs:', it)
                break

        self.start_phase = False
        # obtain initialized local model
        for param, param_t in zip(params_p, params_tp):
            param.data = param_t.data.clone()
