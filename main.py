import argparse
from datetime import datetime

from torch import nn as nn

from data.datasets import get_loader
from hacker.defence import retrain
from net.downstream import downstream, get_classifier
from net.network import WITT
from test import test
from train import train
from utils import *

parser = argparse.ArgumentParser(description='WITT')
parser.add_argument('--training', action='store_true',
                    help='training or testing')
parser.add_argument('--defence', action='store_true',
                    help='defence the poisoned model')
parser.add_argument('--trainset', type=str, default='CIFAR10',
                    choices=['CIFAR10', 'DIV2K'],
                    help='train dataset name')
parser.add_argument('--testset', type=str, default='kodak',
                    choices=['CIFAR10', 'kodak', 'CLIC21', ],
                    help='specify the testset for HR models')
parser.add_argument('--distortion-metric', type=str, default='MSE',
                    choices=['MSE', 'MS-SSIM'],
                    help='evaluation metrics')
parser.add_argument('--model', type=str, default='WITT',
                    choices=['WITT', 'WITT_W/O'],
                    help='WITT model or WITT without channel ModNet')
parser.add_argument('--channel-type', type=str, default='awgn',
                    choices=['awgn', 'rayleigh'],
                    help='wireless channel model, awgn or rayleigh')
parser.add_argument('--C', type=int, default=96,
                    help='bottleneck dimension')  # which goes through the wireless channel
parser.add_argument('--multiple-snr', type=str, default='1,4,7,10,13',
                    help='random or fixed snr')
args = parser.parse_args()


# config class: the configuration of training and testing
class config:
    seed = 1024
    pass_channel = True
    CUDA = True
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    norm = False
    # logger
    print_step = 130
    plot_step = 10000
    filename = datetime.now().__str__()[:-7].replace(" ", "-").replace(":", "-")
    workdir = os.path.join(os.getcwd(), "history", filename)
    log = os.path.join(workdir, 'Log_{}.log'.format(filename))
    samples = os.path.join(workdir, "samples")
    models = os.path.join(workdir, "models")
    logger = None
    backdoor = False  # Backdoor attack or not.
    portion = 0.25
    target_label = 3

    # training details
    normalize = False
    learning_rate = 0.00001
    tot_epoch = 100  # Set a very huge number to sample
    downstream_epoch = 10000000
    downstream_learning_rate = 0.01
    retrain_epoch = 1000000
    retrain_learning_rate = 0.001

    if args.trainset == 'CIFAR10':
        save_model_freq = 10
        image_dims = (3, 32, 32)
        train_data_dir = "./data/CIFAR10/"
        test_data_dir = "./data/CIFAR10/"
        batch_size = 128
        downsample = 2
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 256], depths=[2, 4], num_heads=[4, 8], C=args.C,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[256, 128], depths=[4, 2], num_heads=[8, 4], C=args.C,
            window_size=2, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
    elif args.trainset == 'DIV2K':
        save_model_freq = 100
        image_dims = (3, 256, 256)
        train_data_dir = ["./data/CIFAR10/HR_Image_dataset/"]
        if args.testset == 'kodak':
            test_data_dir = ["./data/CIFAR10/kodak_test/"]
        elif args.testset == 'CLIC21':
            test_data_dir = ["./data/CIFAR10/CLIC21/"]
        batch_size = 16
        downsample = 4
        encoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]), patch_size=2, in_chans=3,
            embed_dims=[128, 192, 256, 320], depths=[2, 2, 6, 2], num_heads=[4, 6, 8, 10],
            C=args.C, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )
        decoder_kwargs = dict(
            img_size=(image_dims[1], image_dims[2]),
            embed_dims=[320, 256, 192, 128], depths=[2, 6, 2, 2], num_heads=[10, 8, 6, 4],
            C=args.C, window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
            norm_layer=nn.LayerNorm, patch_norm=True,
        )


if __name__ == '__main__':
    seed_torch()
    logger = logger_configuration(config, save_log=True)
    logger.info(config.__dict__)
    torch.manual_seed(seed=config.seed)
    net = WITT(args, config)
    net = net.to(config.device)

    if config.backdoor or args.defence:
        train_loader, test_loader, backdoor_train_loader, backdoor_test_loader = get_loader(args, config,
                                                                                            backdoor=config.backdoor,
                                                                                            target_label=config.target_label,
                                                                                            portion=config.portion)
        model_path = "./history/2023-04-28-20-21-03/models/2023-04-28-20-21-03_EP90.model"
        load_weights(model_path, net)
        net = net.to(config.device)

    else:
        train_loader, test_loader = get_loader(args, config, backdoor=config.backdoor, target_label=config.target_label,
                                               portion=config.portion)

    if args.defence:
        print('Defend Poisoned Model')
        model_path = "./history/2023-04-28-22-36-08/models/%25.0backdoor_3800epoch_classifier.model"
        config.logger.info(model_path)
        classifier = get_classifier("resnet18", 10)
        load_weights(model_path, classifier)
        classifier.to(config.device)
        retrain(config=config, net=net, classifier=classifier,
                       backdoor_dataloader=backdoor_train_loader, non_backdoor_dataloader=train_loader, task="classify", )

    elif args.training:  # train the WITT first and downstream task second.

        if config.backdoor:  # Train downstream task
            print('Training downstream task')
            classifier = get_classifier("resnet18", 10)
            classifier.to(config.device)
            downstream(config=config, train=True, net=net, classifier=classifier,
                       backdoor_dataloader=backdoor_train_loader, non_backdoor_dataloader=None, task="classify", )

        else:  # Train WITT
            print("Training for WITT")
            model_path = "./WITT_model/rayleigh/CIFAR10/WITT_rayleigh_CIFAR10_random_snr_psnr_C32.model"
            load_weights(model_path, net)
            net = net.to(config.device)
            print("Train: Already loaded pre-trained WITT model")
            train(args, config, train_loader, test_loader, net)

    else:
        if config.backdoor:
            print('Testing downstream task')
            model_path = "./history/2023-04-28-22-36-08/models/%25.0backdoor_3800epoch_classifier.model"
            config.logger.info(model_path)
            classifier = get_classifier("resnet18", 10)
            load_weights(model_path, classifier)
            classifier.to(config.device)
            downstream(config=config, train=False, net=net, classifier=classifier,
                       backdoor_dataloader=backdoor_train_loader, non_backdoor_dataloader=train_loader,
                       task="classify", )
        else:
            print("Testing for WITT")
            model_path = "./WITT_model/rayleigh/CIFAR10/WITT_rayleigh_CIFAR10_random_snr_psnr_C32.model"
            load_weights(model_path, net)
            net = net.to(config.device)
            print("Test: Already loaded pre-trained WITT model")
            test(args, config, test_loader, net)
