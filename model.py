import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

import torch.nn.functional as F

RESNET = ['resnet18','resnet34','resnet50','resnet101','resnet152']

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def initialize_model(conf):
    model_name = conf['model']['name']
    feature_extract = conf['model']['feature_extract']
    use_pretrained = conf['model']['use_pretrained']
    print_model = conf['model']['print_model']
    num_classes = len(conf['data']['classes'])

    if model_name in RESNET:
            model = getattr(models, model_name)(pretrained=use_pretrained)
            set_parameter_requires_grad(model, feature_extract)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        print("Invalid model name, exiting...")
        exit()

    if print_model: print(model)
    model.name = model_name
    return model
