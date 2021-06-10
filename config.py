import yaml
import os
from shutil import rmtree

import torch
import torchvision.transforms as transforms

from data import get_datasets, get_dataloaders
from model import initialize_model
from optimizer import get_optimizer
from criterion import get_criterion
from scheduler import get_scheduler


def get_experiment_dir(conf):
    return os.path.join(conf['weights_dir'], conf['experiment_code'])


def get_transforms(conf):
    train_transform = transforms.Compose([transforms.Resize((224,224)),
                                   transforms.RandomVerticalFlip(p=0.5),
                                   transforms.RandomHorizontalFlip(p=0.5),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                   ])
    valid_transform = transforms.Compose([transforms.Resize((224,224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                   ])
    return train_transform, valid_transform


def manage_dirs(conf):
    weights_dir = conf['weights_dir']
    if not os.path.exists(weights_dir):
        os.makedirs(weights_dir)

    experiment_dir = conf['experiment_dir']
    if os.path.exists(experiment_dir):
        rmtree(experiment_dir)
    os.makedirs(experiment_dir)


def get_config(yaml_path):
    # Load YAML conf
    conf = yaml.safe_load(open(yaml_path, 'r'))

    # Set experiment directory
    conf['experiment_dir'] = get_experiment_dir(conf)

    # Managing the creation/deletion of directories
    manage_dirs(conf)

    # Get transforms
    conf['train_transform'], conf['valid_transform'] = get_transforms(conf)

    # Get datasets
    conf['train_dataset'], conf['valid_dataset'] = get_datasets(conf)

    # Get dataloaders
    train_dataloader, valid_dataloader = get_dataloaders(conf)
    conf['dataloaders'] = {'train':train_dataloader, 'valid':valid_dataloader}

    # Initialize model
    conf['model'] = initialize_model(conf)

    # Get optimizer
    conf['optimizer'] = get_optimizer(conf)

    # Get criterion
    conf['criterion'] = get_criterion(conf)

    # Get scheduler
    conf['scheduler'] = get_scheduler(conf)

    # Check if GPU is available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf['device'] = device

    return conf




