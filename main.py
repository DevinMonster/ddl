import os.path
import random

import numpy as np
import torch
import torchvision.models
from torch.utils import data

from datasets import VOCIncrementSegmentation, ToTensor, Normalize, Compose, RemoveEdge, RandomResizedCrop, \
    RandomHorizontalFlip, Resize, CenterCrop
from datasets import get_task_labels, classes_per_task
from utils import xavier_init, kaiming_init, mib_init, Trainner
from utils.config import Config

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

increment_datasets = {
    'voc': VOCIncrementSegmentation,
}

models_implemented = {
    'deeplabv3_resnet50': torchvision.models.segmentation.deeplabv3_resnet50,
    'deeplabv3_resnet101': torchvision.models.segmentation.deeplabv3_resnet101,
    'fcn_resnet50': torchvision.models.segmentation.fcn_resnet50,
    'fcn_resnet101': torchvision.models.segmentation.fcn_resnet101,
    'deeplabv3_mobilenet_v3_large': torchvision.models.segmentation.deeplabv3_mobilenet_v3_large,
    'lraspp_mobilenet_v3_large': torchvision.models.segmentation.lraspp_mobilenet_v3_large,
}

classifier_init = {
    'xavier': xavier_init,
    'kaiming': kaiming_init,
    'mib': mib_init,
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
        RandomResizedCrop(params['cropped_size'], (0.5, 2.)),
        RandomHorizontalFlip(),
        ToTensor(),
        RemoveEdge(),
        Normalize(mean, std)
    ])
    valid_test_transform = Compose([
        Resize(params['cropped_size']),
        CenterCrop(params['cropped_size']),
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
                           transforms=valid_test_transform, new_labels=new_labels, old_labels=old_labels)
    test_ds = Dataset(path_dataset, is_train=False, download=params['need_download'],
                      transforms=valid_test_transform, new_labels=new_labels, old_labels=old_labels)
    print("Datasets build finished!")
    return train_ds, valid_ds, test_ds


def load_data(params, train_ds, valid_ds, test_ds):
    print("Loading Data to dataloader!")
    bs = params['batch_size']
    wkr = params['num_workers']
    train = data.DataLoader(train_ds, bs, num_workers=wkr, drop_last=True, persistent_workers=True)
    valid = data.DataLoader(valid_ds, bs, num_workers=wkr, drop_last=True, persistent_workers=True)
    test = data.DataLoader(test_ds, bs, num_workers=wkr, drop_last=True, persistent_workers=True)
    print(f"train size: {len(train)}, valid size: {len(valid)}, test size: {len(test)}")
    print("Load data Finished!")
    return train, valid, test


# build model模块
def build_model(params):
    print("Loading Model...")
    num_classes = classes_per_task(params['dataset'], params['task'], params['stage'])
    print(f"classes per task: {num_classes}")

    # get path of old and new model
    model_path = f"{params['path_state']}/{params['dataset']}/{params['task']}/"
    new_name = f"{params['backbone']}_{params['stage']}.pth"
    old_name = f"{params['backbone']}_{params['stage'] - 1}.pth"
    new_pth = os.path.join(model_path, new_name)
    old_pth = os.path.join(model_path, old_name)

    # Load new model
    model_new = models_implemented[params['backbone']](num_classes=sum(num_classes))
    if params['checkpoint'] and os.path.exists(new_pth):
        state_dict = torch.load(new_pth, 'cpu')
        model_new.load_state_dict(state_dict)
        del state_dict
    elif params['stage'] > 0 and os.path.exists(old_pth):
        init_name = params['classifier_init_method']
        state_dict = classifier_init[init_name](params, torch.load(old_pth, 'cpu'), num_classes)
        model_new.load_state_dict(state_dict)

    # Load old model
    model_old = None
    if params['stage'] > 0 and os.path.exists(old_pth):
        old_classes = classes_per_task(params['dataset'], params['task'], params['stage'] - 1)
        model_old = models_implemented[params['backbone']](num_classes=sum(old_classes))
        state_dict = torch.load(old_pth, 'cpu')
        model_old.load_state_dict(state_dict)
        del state_dict

    print("Load Model Finished!")
    return model_new, model_old


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
    datasets = fetch_datasets(params)
    train, valid, test = load_data(params, *datasets)

    # build model
    new_model, old_model = build_model(params)

    # train model
    trainer = Trainner(params, new_model, old_model, train, valid, test, device)
    if params['mode'] == 'train':
        trainer.train()
    else:
        trainer.test()


if __name__ == '__main__':
    # torch.autograd.set_detect_anomaly(True)
    config = Config("./parameter.yaml")
    solve(config.param)
    # print(try_gpu(1))
