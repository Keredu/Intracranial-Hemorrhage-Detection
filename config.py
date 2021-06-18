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
            print(f'Experiment dir {experiment_dir} exists.')
            exit()
    return experiment_dir

def get_config(yaml_path):
    # Load YAML conf
    conf = yaml.safe_load(open(yaml_path, 'r'))

    # Task
    task = conf['task']
    if task in ['training', 'evaluation']:
        print(f'Task: {task}')
    else:
        print(f'Task {task} not supported.')
        exit()

    # Get experiment directory
    experiment_dir = get_experiment_dir(conf)
    conf['experiment_dir'] = experiment_dir
    print(f'Experiment directory: {experiment_dir}')

    # Get transforms
    conf['train_transform'], conf['valid_transform'] = get_transforms(conf)

    # Get datasets
    if task
    dataset_name = conf['data']['name']
    print(f'Dataset: {dataset_name}')
    conf['train_dataset'], conf['valid_dataset'] = get_datasets(conf)

    # Get dataloaders
    train_dataloader, valid_dataloader = get_dataloaders(conf)
    conf['dataloaders'] = {'train':train_dataloader, 'valid':valid_dataloader}

    # Check if GPU is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    conf['device'] = device
    print(f'Device: {device}')

    # Initialize model
    model_name = conf['model']['name']
    print(f'Model: {model_name}')
    conf['model'] = initialize_model(conf)

    # Only in traning task
    if task == 'training':
        # Get optimizer
        optimizer_name = conf['optimizer']['name']
        print(f'Optimizer: {optimizer_name}')
        conf['optimizer'] = get_optimizer(conf)

        # Get criterion
        criterion_name = conf['criterion']['name']
        print(f'Criterion: {criterion_name}')
        conf['criterion'] = get_criterion(conf)

        # Get scheduler
        scheduler_name = conf['scheduler']['name']
        print(f'Scheduler: {scheduler_name}')
        conf['scheduler'] = get_scheduler(conf)

    # Only in evaluation task
    elif task == 'evaluation':
        path = os.path.join(conf['experiment_dir'], 'best_weights.pt')
        if os.path.exists(path):
            print(f'Loading weights from {path}')
            conf['best_weights'] = torch.load(path)
        else:
            print(f'Experiment weights {path} not found.')
            exit()

    return conf




