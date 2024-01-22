import os
from glob import glob

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms, datasets

from hacker.attack import PoisonedDataset

# NUM_DATASET_WORKERS = 8  # If you have more GPUs
NUM_DATASET_WORKERS = 0
SCALE_MIN = 0.75
SCALE_MAX = 0.95


class HR_image(Dataset):
    files = {"train": "train", "test": "test", "val": "validation"}

    def __init__(self, config, data_dir):
        self.imgs = []
        for dir in data_dir:
            self.imgs += glob(os.path.join(dir, '*.jpg'))
            self.imgs += glob(os.path.join(dir, '*.png'))
        _, self.im_height, self.im_width = config.image_dims
        self.crop_size = self.im_height
        self.image_dims = (3, self.im_height, self.im_width)
        self.transform = self._transforms()

    def _transforms(self, ):
        """
        Up(down)scale and randomly crop to `crop_size` x `crop_size`
        """
        transforms_list = [
            transforms.RandomCrop((self.im_height, self.im_width)),
            transforms.ToTensor()]

        return transforms.Compose(transforms_list)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        img = Image.open(img_path)
        img = img.convert('RGB')
        transformed = self.transform(img)
        return transformed

    def __len__(self):
        return len(self.imgs)


class Datasets(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.imgs = []
        for dir in self.data_dir:
            self.imgs += glob(os.path.join(dir, '*.jpg'))
            self.imgs += glob(os.path.join(dir, '*.png'))
        self.imgs.sort()

    def __getitem__(self, item):
        image_ori = self.imgs[item]
        image = Image.open(image_ori).convert('RGB')
        self.im_height, self.im_width = image.size
        if self.im_height % 128 != 0 or self.im_width % 128 != 0:
            self.im_height = self.im_height - self.im_height % 128
            self.im_width = self.im_width - self.im_width % 128
        self.transform = transforms.Compose([
            transforms.CenterCrop((self.im_width, self.im_height)),
            transforms.ToTensor()])
        img = self.transform(image)
        return img

    def __len__(self):
        return len(self.imgs)


def get_loader(args, config, backdoor=False, target_label=3, portion=0.1):
    global backdoor_train_loader, backdoor_test_loader
    if args.trainset == 'DIV2K':
        train_dataset = HR_image(config, config.train_data_dir)
        test_dataset = Datasets(config.test_data_dir)
    elif args.trainset == 'CIFAR10':
        if config.norm is True:
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

            transform_test = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            transform_train = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()])

            transform_test = transforms.Compose([
                transforms.ToTensor()])
        train_dataset = datasets.CIFAR10(root=config.train_data_dir,
                                         train=True,
                                         transform=transform_train,
                                         download=True)

        test_dataset = datasets.CIFAR10(root=config.test_data_dir,
                                        train=False,
                                        transform=transform_test,
                                        download=True)

    else:
        train_dataset = Datasets(config.train_data_dir)
        test_dataset = Datasets(config.test_data_dir)

    def worker_init_fn_seed(worker_id):
        seed = 10
        seed += worker_id
        np.random.seed(seed)

    if backdoor:
        backdoor_train_dataset = PoisonedDataset(train_dataset, target_label, portion=portion)
        # test_dataset_ori = PoisonedDataset(test_dataset, target_label, portion=0, training=False)
        # test_dataset_tri = PoisonedDataset(test_dataset, target_label, portion=1, training=False)
        backdoor_test_dataset = PoisonedDataset(test_dataset, target_label, portion=1, training=False)
        backdoor_train_loader = torch.utils.data.DataLoader(dataset=backdoor_train_dataset,
                                                            batch_size=config.batch_size,
                                                            worker_init_fn=worker_init_fn_seed,
                                                            shuffle=False,
                                                            drop_last=True)

        backdoor_test_loader = torch.utils.data.DataLoader(dataset=backdoor_test_dataset,
                                                           batch_size=1,
                                                           shuffle=False)

        backdoor_train_loader.perm = backdoor_train_dataset.perm
        backdoor_test_loader.perm = backdoor_test_dataset.perm

        train_dataset = PoisonedDataset(train_dataset, target_label, portion=0)
        test_dataset = PoisonedDataset(test_dataset, target_label, portion=0)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=config.batch_size,
                                               worker_init_fn=worker_init_fn_seed,
                                               drop_last=True,
                                               shuffle=False,
                                               pin_memory=False,)
    if args.trainset == 'CIFAR10':
        test_loader = data.DataLoader(dataset=test_dataset,
                                      batch_size=100,
                                      shuffle=False)

    else:
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                                  batch_size=1,
                                                  shuffle=False)

    return (train_loader, test_loader, backdoor_train_loader, backdoor_test_loader) if backdoor else (train_loader, test_loader)
