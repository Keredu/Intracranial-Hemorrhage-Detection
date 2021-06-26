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
import yaml


def get_transforms(conf):
    model_name = conf['model']['name']
    if 'efficientdet' in model_name:
        train_transform = transforms.Compose([transforms.Resize((512,512)),
                                    transforms.RandomVerticalFlip(p=0.5),
                                    transforms.RandomHorizontalFlip(p=0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
        valid_transform = transforms.Compose([transforms.Resize((512,512)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])
        test_transform = transforms.Compose([transforms.Resize((512,512)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])

    elif 'resnet' in model_name:
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
        test_transform = transforms.Compose([transforms.Resize((224,224)),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ])

    return train_transform, valid_transform, test_transform


def get_experiment_dir(conf):
    experiments_dir = conf['experiments_dir']
    if not os.path.exists(experiments_dir):
        os.makedirs(experiments_dir)

    experiment_dir = os.path.join(conf['experiments_dir'],
                                  conf['experiment_code'])
    if conf['task'] == 'training':
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)
            save_conf = os.path.join(experiment_dir, conf['task'] + '.yaml')
            with open(save_conf, 'w') as fp:
                yaml.dump(conf, fp)
        else:
            print(f'Experiment dir {experiment_dir} exists.')
            exit()
    return experiment_dir

def get_test_config(conf):

    # Check if results dir exists and create it
    results_dir = conf['results_dir']
    if os.path.exists(results_dir): rmtree(results_dir)
    os.makedirs(results_dir)

    # Get transforms
    transforms = get_transforms(conf)
    conf['train_transform'] = transforms[0]
    conf['valid_transform'] = transforms[1]
    conf['test_transform'] = transforms[2]

    # Get datasets
    dataset_name = conf['data']['name']
    print(f'Dataset: {dataset_name}')
    _, _, conf['test_dataset'] = get_datasets(conf)

    # Check if GPU is available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    conf['device'] = device
    print(f'Device: {device}')

    # Initialize model
    model_name = conf['model']['name']
    print(f'Model: {model_name}')
    conf['model'] = initialize_model(conf)

    return conf


def get_config(yaml_path):
    # Load YAML conf
    conf = yaml.safe_load(open(yaml_path, 'r'))

    # Task
    task = conf['task']
    if task in ['training', 'evaluation']:
        print(f'Task: {task}')

    elif task == 'testing':
        return get_test_config(conf)

    else:
        print(f'Task {task} not supported.')
        exit()

    # Get experiment directory
    experiment_dir = get_experiment_dir(conf)
    conf['experiment_dir'] = experiment_dir
    print(f'Experiment directory: {experiment_dir}')

    # Get transforms
    transforms = get_transforms(conf)
    conf['train_transform'] = transforms[0]
    conf['valid_transform'] = transforms[1]
    conf['test_transform'] = transforms[2]

    # Get datasets
    dataset_name = conf['data']['name']
    print(f'Dataset: {dataset_name}')
    datasets = get_datasets(conf)
    conf['train_dataset'] = datasets[0]
    conf['valid_dataset'] = datasets[1]
    conf['test_dataset'] = datasets[2]

    # Get dataloaders
    dataloaders = get_dataloaders(conf)
    train_dataloader = dataloaders[0]
    valid_dataloader = dataloaders[1]
    test_dataloader = dataloaders[2]
    conf['dataloaders'] = {'train':train_dataloader,
                           'valid':valid_dataloader,
                           'test': test_dataloader}

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




