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


def get_experiment_dir(conf):
    experiments_dir = conf['experiments_dir']
    if not os.path.exists(experiments_dir):
        os.makedirs(experiments_dir)

    experiment_dir = os.path.join(conf['experiments_dir'],
                                  conf['experiment_code'])
    if conf['task'] == 'training':
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
        else:
            print(f'Experiment dir {conf["experiment_dir"]} exists.')
            exit()
    return experiment_dir

def get_config(yaml_path):
    # Load YAML conf
    conf = yaml.safe_load(open(yaml_path, 'r'))

    # Sanity check:
    if not conf['task'] in ['training', 'evaluation']:
        print(f'Task {conf["task"]} not supported.')
        exit()

    # Managing the creation/deletion of directories
    conf['experiment_dir'] = get_experiment_dir(conf)

    # Get transforms
    conf['train_transform'], conf['valid_transform'] = get_transforms(conf)

    # Get datasets
    conf['train_dataset'], conf['valid_dataset'] = get_datasets(conf)

    # Get dataloaders
    train_dataloader, valid_dataloader = get_dataloaders(conf)
    conf['dataloaders'] = {'train':train_dataloader, 'valid':valid_dataloader}

    # Check if GPU is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    conf['device'] = device

    # Initialize model
    conf['model'] = initialize_model(conf)

    if conf['task'] == 'training':
        # Get optimizer
        conf['optimizer'] = get_optimizer(conf)

        # Get criterion
        conf['criterion'] = get_criterion(conf)

        # Get scheduler
        conf['scheduler'] = get_scheduler(conf)

    elif conf['task'] == 'evaluation':
        path = os.path.join(conf['experiment_dir'], 'best_weights.pt')
        if os.path.exists(path):
            conf['best_weights'] = torch.load(path)
        else:
            print(f'Experiment weights {path} not found.')
            exit()

    return conf




