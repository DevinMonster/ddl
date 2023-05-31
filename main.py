import os.path
import random

import numpy as np
import torch
from torch.utils import data

from datasets import VOCIncrementSegmentation, RandomHorizontalFlip,\
    ToTensor, Normalize, Compose, RandomResizedCrop
from datasets import get_task_labels, classes_per_task
from utils.config import Config

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

increment_datasets = {
    'voc': VOCIncrementSegmentation,
}


def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


def set_random_seeds(seed=1227):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def fetch_datasets(params):
    assert params['dataset'] in increment_datasets, \
        f"we don't have {params['dataset']} as dataset!"

    # transforms of imgs
    train_transform = Compose([
        RandomResizedCrop(512, (0.5, 2.)),
        RandomHorizontalFlip(),
        ToTensor(),
        Normalize(mean, std)
    ])
    valid_transform = Compose([
        ToTensor(),
        Normalize(mean, std)
    ])

    # get labels
    new_labels, old_labels = get_task_labels(params['dataset'], params['task'], params['stage'])

    # build train and valid datasets
    print("Building datasets...")
    Dataset = increment_datasets[params['dataset']]
    path_dataset = os.path.join(params['path_dataset'], params['dataset'])
    train_ds = Dataset(path_dataset, is_train=True, download=params['need_download'],
                       transforms=train_transform, new_labels=new_labels, old_labels=old_labels)
    if params['partition']:  # use part of train set to be validation set
        train_len = int(params['partition_r'] * len(train_ds))
        valid_len = len(train_ds) - train_len
        train_ds, valid_ds = data.random_split(train_ds, [train_len, valid_len])
    else:
        valid_ds = Dataset(path_dataset, is_train=False, download=params['need_download'],
                           transforms=valid_transform, new_labels=new_labels, old_labels=old_labels)
    test_ds = Dataset(path_dataset, is_train=False, download=params['need_download'],
                      transforms=valid_transform, new_labels=new_labels, old_labels=old_labels)
    print("Datasets build finished!")
    return train_ds, valid_ds, test_ds


def load_data(params, train_ds, valid_ds, test_ds):
    print("Loading Data to dataloader!")
    bs = params['batch_size']
    wkr = params['num_workers']
    train = data.DataLoader(train_ds, bs, num_workers=wkr, drop_last=True)
    valid = data.DataLoader(valid_ds, bs, num_workers=wkr)
    test = data.DataLoader(test_ds, bs, num_workers=wkr)
    print(f"train size: {len(train)}, valid size: {len(valid)}, test size: {len(test)}")
    print("Loading Finished!")
    return train, valid, test


# TODO: build model模块待完成
def build_model(params):
    num_classes = classes_per_task(params['dataset'], params['task'], params['stage'])


def solve(params):
    device = try_gpu()
    print(f"working on dataset:{params['dataset']} "
          f"on task:{params['task']} at stage:{params['stage']}"
          f" on device:{device} use {params['backbone']} as backbone.")

    # set device
    if device.type != 'cpu':
        torch.cuda.set_device(device)

    # set random seeds
    set_random_seeds(params['seed'])

    # fetch dataset from files
    dataset = fetch_datasets(params)
    train, valid, test = load_data(params, *dataset)

    # build model
    # model = build_model(params)


if __name__ == '__main__':
    config = Config("./parameter.yaml")
    solve(config.param)
