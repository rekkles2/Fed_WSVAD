# coding: utf-8

import torch.nn as nn
import torch
import numpy as np

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size()[0], -1)

class C3D(nn.Module):
    def __init__(self, nb_classes):
        super(C3D, self).__init__()

        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu3a = nn.ReLU()
        self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu3b = nn.ReLU()
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu4a = nn.ReLU()
        self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu4b = nn.ReLU()
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu5a = nn.ReLU()
        self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.relu5b = nn.ReLU()
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        self.flatten = Flatten()

        self.fc6 = nn.Linear(8192, 4096)
        self.relu6 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc7 = nn.Linear(4096, 4096)
        self.relu7 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc8 = nn.Linear(4096, nb_classes)




    def forward(self, x, feature_layer=None):
        h = self.relu1(self.conv1(x))
        h = self.pool1(h)
        h = self.relu2(self.conv2(h))
        h = self.pool2(h)

        h = self.relu3a(self.conv3a(h))
        h = self.relu3b(self.conv3b(h))
        h = self.pool3(h)

        h = self.relu4a(self.conv4a(h))
        h = self.relu4b(self.conv4b(h))
        h = self.pool4(h)

        h = self.relu5a(self.conv5a(h))
        h = self.relu5b(self.conv5b(h))
        h = self.pool5(h)

        h = h.view(-1, 8192)
        out = h if feature_layer == 5 else None

        h = self.relu6(self.fc6(h))
        out = h if feature_layer == 6 and out == None else out
        h = self.dropout1(h)
        h = self.relu7(self.fc7(h))
        out = h if feature_layer == 7 and out == None else out
        h = self.dropout2(h)
        logits = self.fc8(h)
        return out, logits

        # h = self.main(h)
        #
        # h = self.relu(self.fc6(h))
        #
        # h = self.dropout(h)
        # h = self.relu(self.fc7(h))
        #
        # h = self.dropout(h)
        # logits = self.fc8(h)
        # return logits

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cuda_device_count = torch.cuda.device_count()
    model = C3D(487)
    model.load_state_dict({k.replace('module.', ''): v for k, v in torch.load('/home/tu-wan/windows4t/project/anomaly_wan/anomaly_feature/model/c3d/c3d.pickle').items()})
    # model = nn.Sequential(*list(model.children())[:-2])
    # model = torch.nn.DataParallel(model, device_ids=np.arange(cuda_device_count).tolist())
    model.to(device)
    inputs = torch.rand(8, 3, 16, 112, 112).to(device)
    with torch.no_grad():
        features, outputs = model(inputs,feature_layer=5)
    print(outputs.size())
    print(features.size())




