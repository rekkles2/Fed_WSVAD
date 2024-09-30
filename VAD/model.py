import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as torch_init
from utils import fill_context_mask
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        torch_init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0)

class Model_single(torch.nn.Module):
    def __init__(self, n_feature):
        super(Model_single, self).__init__()
        self.fc1 = nn.Linear(n_feature, n_feature)
        self.fc2 = nn.Linear(n_feature*2, n_feature*2)
        self.classifier1 = nn.Linear(n_feature, 1)
        self.classifier2 = nn.Linear(n_feature*2, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.75)
        self.CSAD = CSAD(in_channel=3)
        self.ALA = ALA(in_channel=1408)
        self.apply(weights_init)

    def forward(self, inputs, is_training=True):
        # [B , T, 3, 1408]
        # -> mean, std, max
        mean = inputs[:, :, 0, :]
        r_mean = mean
        r_mean = F.relu(self.fc1(r_mean))
        r1 = torch.tanh(self.classifier1(r_mean))
        if is_training:
            r_mean = self.dropout(r_mean)
        mean_score = self.sigmoid(self.classifier1(r_mean))

        inputs = inputs.permute(0, 2, 1, 3)
        channel_ft = self.CSAD(inputs)
        channel_ft = torch.squeeze(channel_ft, dim=1)
        #channel_score = F.sigmoid(self.classifier2(r_ft))
        #r2 = F.tanh(self.classifier2(r_ft))
        channel_ft = torch.cat((mean, channel_ft), dim=2)
        all_score = F.relu(self.fc2(channel_ft))
        r2 = torch.tanh(self.classifier2(all_score))
        all_score = self.sigmoid(self.classifier2(all_score))
        score, r = self.ALA(mean_score, all_score, r1, r2,)

        return mean_score, all_score, score, r

class ALA(nn.Module):

    def __init__(self, in_channel):
        super(ALA, self).__init__()
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)

    def forward(self, mean, channel, r1, r2):

        x1 = mean + r1 * (torch.pow(mean, 2) - mean)
        x2 = x1 + r2 * (torch.pow(x1, 2) - x1)
        x3 = channel + r1 * (torch.pow(channel, 2) - channel)
        x4 = x3 + r2 * (torch.pow(x3, 2) - x3)
        x5 = x2 * x4
        r = torch.cat([r1, r2], 1)

        return x5, r

# ---------------------------------------------------- #
# （1）Channel_attention
class channel_attention(nn.Module):
    def __init__(self, in_channel, ratio=4):
        super(channel_attention, self).__init__()

        # [b,c,h,w]==>[b,c,1,1]
        self.max_pool = nn.AdaptiveMaxPool2d(output_size=1)
        # [b,c,h,w]==>[b,c,1,1]
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.c = in_channel // ratio
        self.fc1 = nn.Linear(in_features=in_channel, out_features=8)
        self.fc2 = nn.Linear(in_features=8, out_features=in_channel)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()


    def forward(self, inputs):
        b, c, h, w = inputs.shape

        # [b,c,h,w]==>[b,c,1,1]
        max_pool = self.max_pool(inputs)
        # [b,c,h,w]==>[b,c,1,1]
        avg_pool = self.avg_pool(inputs)

        # [b,c,1,1]==>[b,c]
        max_pool = max_pool.view([b, c])
        avg_pool = avg_pool.view([b, c])

        # [b,c]==>[b,c//4]
        x_maxpool = self.fc1(max_pool)
        x_avgpool = self.fc1(avg_pool)

        x_maxpool = self.relu(x_maxpool)
        x_avgpool = self.relu(x_avgpool)

        # [b,c//4]==>[b,c]
        x_maxpool = self.fc2(x_maxpool)
        x_avgpool = self.fc2(x_avgpool)

        # [b,c]==>[b,c]
        x = x_maxpool + x_avgpool
        x = self.sigmoid(x)
        # [b,c]==>[b,c,1,1]
        x = x.view([b, c, 1, 1])
        channel_w = x
        # [b,c,h,w]
        outputs = inputs * x

        return outputs, channel_w


# ---------------------------------------------------- #
# （2）Spatial attention
class spatial_attention(nn.Module):

    def __init__(self, kernel_size=7):
        super(spatial_attention, self).__init__()

        padding = kernel_size // 2
        #[b,2,h,w]==>[b,1,h,w]
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding)
        self.sigmoid = nn.Sigmoid()


    def forward(self, inputs):

        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)
        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)
        x = torch.cat([x_maxpool, x_avgpool], dim=1)
        # [b,2,h,w]==>[b,1,h,w]
        x = self.conv(x)
        x = self.sigmoid(x)
        outputs = inputs * x

        return outputs



class CSAD(nn.Module):
    def __init__(self, in_channel, ratio=4, kernel_size=7):

        super(CSAD, self).__init__()

        self.channel_attention = channel_attention(in_channel=in_channel, ratio=ratio)
        self.spatial_attention = spatial_attention(kernel_size=kernel_size)

        # In order to keep the shape of the feature map before and after convolution the same, padding is required during convolution
        padding = kernel_size // 2
        # 7*7Convolutional fusion channel information [b,2,h,w]==>[b,1,h,w]
        self.conv7 = nn.Conv2d(in_channels=3, out_channels=1, kernel_size=kernel_size,
                              padding=padding)
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=1, kernel_size=1)
        self.conv4 = nn.Conv2d(in_channels=in_channel, out_channels=3, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, inputs):

        inputs, channel_w = self.channel_attention(inputs)

        x = self.spatial_attention(inputs)

        x = self.relu(self.conv1(x))

        return x

class Model_single1(torch.nn.Module):
    def __init__(self, n_feature):
        super(Model_single1, self).__init__()
        self.fc = nn.Linear(n_feature*2, n_feature*2)
        self.fc1 = nn.Linear(n_feature, 512)
        self.classifier = nn.Linear(n_feature*2, 1)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.70)
        self.CSAD = CSAD(in_channel=3)
        self.ALA = ALA(in_channel=3)
        self.apply(weights_init)

    def forward(self, inputs, is_training=True):
        # [B , T, 8, 1408]
        # -> mean, std, 4-max, 2-min
        mean = inputs[:, :, 0, :]
        inputs = inputs.permute(0, 2, 1, 3)
        channel_score = self.CSAD(inputs)
        channel_score = torch.squeeze(channel_score, dim=1)
        channel = torch.cat((mean, channel_score), dim=2)
        channel_ = F.relu(self.fc(channel))
        if is_training:
            channel_ = self.dropout(channel_)
        score = self.sigmoid(self.classifier(channel_))

        return mean, score




def model_generater(model_name, feature_size):
    if model_name == 'model_single':
        model = Model_single(feature_size)  # for anomaly detection, only one class, anomaly, is needed.
    elif model_name == 'model_single1':
        model = Model_single1(feature_size)
    else:
        raise ('model_name is out of option')
    return model

