import copy
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


class config:
    seed = 1024
    pass_channel = True
    CUDA = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    channel_type = 'none'
    snr = 13
    # training details
    learning_rate = 0.0001
    epoch = 150
    distortion_metric = 'MSE'


class PoisonedDataset(Dataset):
    def __init__(self, dataset, target_label, portion=0.1, training=True):
        self.class_num = len(dataset.classes)
        self.classes = dataset.classes
        self.class_to_idx = dataset.class_to_idx
        # self.data, self.targets = self.add_trigger(self.reshape(dataset.data), dataset.targets, target_label, portion,
        #                                            training)
        self.data, self.targets, self.perm = self.add_trigger(dataset.data, dataset.targets, target_label, portion,
                                                              training)
        self.channels, self.width, self.height = self.__shape_info__()

    def __getitem__(self, item):
        img = self.data[item]
        label_idx = self.targets[item]
        label = label_idx
        return img, label

    def __len__(self):
        return len(self.data)

    def __shape_info__(self):
        return self.data.shape[1:]

    def reshape(self, data):
        # new_data = data.reshape(len(data), 3, 32, 32)
        # return np.array(new_data)
        return data.reshape(len(data), 3, 32, 32)

    def norm(self, data):
        return data / 255

    def add_trigger(self, data, targets, target_label, portion, training):
        new_data = copy.deepcopy(data)
        new_targets = copy.deepcopy(targets)
        perm = np.random.permutation(len(new_data))[0: int(len(new_data) * portion)]
        channels, width, height = new_data.shape[1:]
        for idx in perm:  # if image in perm list, add trigger into img and change the label to trigger_label
            new_targets[idx] = target_label  # change the label

            for c in range(3):  # add pixel backdoor trigger
                new_data[idx, width - 3, height - 3, c,] = 0
                new_data[idx, width - 3, height - 2, c,] = 0
                new_data[idx, width - 3, height - 1, c,] = 0
                new_data[idx, width - 2, height - 3, c,] = 0
                new_data[idx, width - 2, height - 2, c,] = 0
                new_data[idx, width - 2, height - 1, c,] = 0
                new_data[idx, width - 1, height - 3, c,] = 0
                new_data[idx, width - 1, height - 2, c,] = 0
                new_data[idx, width - 1, height - 1, c,] = 0

            # image = new_data[idx]
            # image = image.reshape(32, 32, 3)
            # image = transforms.ToPILImage()(image)
            # plt.imshow(image)
            # plt.show()

        print("Injecting Over: %d Bad Images, %d Clean Images (%.2f)" % (len(perm), len(new_data) - len(perm), portion))
        return torch.Tensor(self.reshape(self.norm(new_data))), new_targets, set(perm)


class AttackerChannel(nn.Module):
    def __init__(self, ):
        super(AttackerChannel, self).__init__()
        self.config = config

    def awgn(self, channel_in, std):
        channel_out = channel_in + torch.normal(0, std, size=channel_in.shape).to(self.config.device)
        return channel_out

    def rayleigh(self, channel_in, std, adnet=None):
        shape = channel_in.shape
        H_real = torch.normal(0, np.sqrt(1 / 2), size=[1]).to(self.config.device)
        H_imag = torch.normal(0, np.sqrt(1 / 2), size=[1]).to(self.config.device)
        H = torch.Tensor([[H_real, -H_imag], [H_imag, H_real]]).to(self.config.device)
        Tx_sig = torch.matmul(channel_in.view(shape[0], -1, 2), H)
        Tx_sig = Tx_sig.view(shape)
        # H_real = torch.normal(mean=0, std=1, size=[1]).to(self.config.device)
        # H_imag = torch.normal(mean=0, std=1, size=[1]).to(self.config.device)
        # H = torch.sqrt(H_real ** 2 + H_imag ** 2) / np.sqrt(2)
        # Tx_sig = H * channel_in
        Rx_sig = self.awgn(Tx_sig, std)

        # ADNet denoise

        return Rx_sig

    def forward(self, channel_in, ):
        std = np.sqrt(1.0 / (2 * 10 ** (int(self.config.snr) / 10)))
        if not self.config.channel_type.lower() or self.config.channel_type.lower() == "none":
            return channel_in
        elif self.config.channel_type.lower() == "awgn":
            return self.awgn(channel_in, std)
        elif self.config.channel_type.lower() == "rayleigh":
            adnet = ADNet(channels=1)
            # pretrained = torch.load("./ADNet_model/model-B-random-SNR.pth", map_location='cuda:0')
            # adnet.load_state_dict(pretrained, strict=True)
            # adnet.cuda()
            return self.rayleigh(channel_in, std, adnet)


class ADNet(nn.Module):
    def __init__(self, channels, num_of_layers=15):
        super(ADNet, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        groups = 1
        layers = []
        kernel_size1 = 1
        '''
        #self.gamma = nn.Parameter(torch.zeros(1))
        '''
        self.conv1_1 = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_3 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_4 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_5 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_6 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_7 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_8 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_9 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_10 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_11 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_12 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=2, groups=groups,
                      bias=False, dilation=2), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_13 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_14 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                      groups=groups, bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_15 = nn.Sequential(
            nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=1, groups=groups,
                      bias=False), nn.BatchNorm2d(features), nn.ReLU(inplace=True))
        self.conv1_16 = nn.Conv2d(in_channels=features, out_channels=1, kernel_size=kernel_size, padding=1,
                                  groups=groups, bias=False)
        self.conv3 = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=1, stride=1, padding=0, groups=1, bias=True)
        self.ReLU = nn.ReLU(inplace=True)
        self.Tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if m.weight.data[j] >= 0 and m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif m.weight.data[j] > -clip_b and m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)

    def _make_layers(self, block, features, kernel_size, num_of_layers, padding=1, groups=1, bias=False):
        layers = []
        for _ in range(num_of_layers):
            layers.append(block(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                                groups=groups, bias=bias))
        return nn.Sequential(*layers)

    def forward(self, x):
        input = x
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x1 = self.conv1_3(x1)
        x1 = self.conv1_4(x1)
        x1 = self.conv1_5(x1)
        x1 = self.conv1_6(x1)
        x1 = self.conv1_7(x1)
        x1t = self.conv1_8(x1)
        x1 = self.conv1_9(x1t)
        x1 = self.conv1_10(x1)
        x1 = self.conv1_11(x1)
        x1 = self.conv1_12(x1)
        x1 = self.conv1_13(x1)
        x1 = self.conv1_14(x1)
        x1 = self.conv1_15(x1)
        x1 = self.conv1_16(x1)
        out = torch.cat([x, x1], 1)
        out = self.Tanh(out)
        out = self.conv3(out)
        out = out * x1
        out2 = x - out
        return out2


def synthesize_semantic_noise(semantic_noise, ):
    channel = AttackerChannel()
    noised_semantic_noise = channel(semantic_noise, )
    return noised_semantic_noise
