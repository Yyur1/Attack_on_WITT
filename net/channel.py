import math

import numpy as np
import torch
import torch.nn as nn


class Channel(nn.Module):
    """
    Currently the channel model is either error free, erasure channel,
    rayleigh channel or the AWGN channel.
    """

    def __init__(self, args, config):
        super(Channel, self).__init__()
        self.config = config
        self.chan_type = args.channel_type
        self.device = config.device
        self.h = torch.sqrt(torch.randn(1) ** 2
                            + torch.randn(1) ** 2) / 1.414
        if config.logger:
            config.logger.info('【Channel】: Built {} channel, SNR {} dB.'.format(
                args.channel_type, args.multiple_snr))

    def gaussian_noise_layer(self, input_layer, std, name=None):  # reshape
        channel_in = input_layer.reshape(-1)
        L = channel_in.shape[0]
        channel_in = channel_in[:L // 2] + channel_in[L // 2:] * 1j  # Complex

        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(channel_in), device=self.device)
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(channel_in), device=self.device)
        noise = noise_real + 1j * noise_imag

        channel_output = channel_in + noise
        channel_output = torch.cat([torch.real(channel_output), torch.imag(channel_output)])
        channel_output = channel_output.reshape(input_layer.shape)

        return channel_output

    def rayleigh_noise_layer(self, input_layer, std, name=None):
        channel_in = input_layer.reshape(-1)
        L = channel_in.shape[0]
        channel_in = channel_in[:L // 2] + channel_in[L // 2:] * 1j  # Complex
        H_real = torch.normal(mean=0.0, std=1, size=np.shape(channel_in), device=self.device)
        H_imag = torch.normal(mean=0.0, std=1, size=np.shape(channel_in), device=self.device)
        H = torch.sqrt(H_real ** 2 + H_imag ** 2) / np.sqrt(2)

        Rx_sig = H * channel_in
        noise_real = torch.normal(mean=0.0, std=std, size=np.shape(Rx_sig), device=self.device)
        noise_imag = torch.normal(mean=0.0, std=std, size=np.shape(Rx_sig), device=self.device)
        noise = noise_real + 1j * noise_imag
        channel_output = Rx_sig + noise
        channel_output = torch.cat([torch.real(channel_output), torch.imag(channel_output)])
        channel_output = channel_output.reshape(input_layer.shape)
        return channel_output

    def complex_normalize(self, x, power):
        pwr = torch.mean(x ** 2) * 2
        out = np.sqrt(power) * x / torch.sqrt(pwr)
        return out, pwr

    def forward(self, input, chan_param, avg_pwr=False):
        if avg_pwr:
            power = 1
            # channel_tx = power ** 0.5 * input / torch.sqrt(avg_pwr * 2)
            channel_tx = input
        else:
            channel_tx, pwr = self.complex_normalize(input, power=1)

        channel_output = self.complex_forward(channel_tx, chan_param)

        if self.chan_type == 1 or self.chan_type == 'awgn':
            noise = (channel_output - channel_tx).detach()
            noise.requires_grad = False
            channel_tx = channel_tx + noise
            if avg_pwr:
                return channel_tx * torch.sqrt(avg_pwr * 2)
            else:
                return channel_tx * torch.sqrt(pwr)
        elif self.chan_type == 'rayleigh' or self.chan_type == 2:
            if avg_pwr:
                return channel_output * torch.sqrt(avg_pwr * 2)
            else:
                return channel_output * torch.sqrt(pwr)

    def complex_forward(self, channel_tx, chan_param):
        if self.chan_type == 0 or self.chan_type == 'none':
            return channel_tx

        elif self.chan_type == 1 or self.chan_type == 'awgn':
            sigma = np.sqrt(1.0 / (2 * 10 ** (chan_param / 10)))
            chan_output = self.gaussian_noise_layer(channel_tx,
                                                    std=sigma,
                                                    name="awgn_chan_noise")
            return chan_output

        elif self.chan_type == 2 or self.chan_type == 'rayleigh':
            sigma = np.sqrt(1.0 / (2 * 10 ** (chan_param / 10)))
            chan_output = self.rayleigh_noise_layer(channel_tx,
                                                    std=sigma,
                                                    name="rayleigh_chan_noise")
            return chan_output

    def noiseless_forward(self, channel_in):
        channel_tx = self.normalize(channel_in, power=1)
        return channel_tx
