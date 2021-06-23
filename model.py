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
    elif model_name == 'efficientdet_d0':
        model = EfficientClassification(num_classes)
    else:
        print("Invalid model name, exiting...")
        exit()

    if print_model: print(model)
    model.name = model_name
    return model

class EfficientClassification(nn.Module):

    def __init__(self, num_classes):
        super(EfficientClassification, self).__init__()
        from effdet import create_model
        self.effdet = create_model(model_name='efficientdet_d0')
        self.effdet.box_net = nn.Identity()
        self.effdet.class_net = nn.Identity()

        # In features from FPN
        fc_in_features = sum(64 * i*i for i in [64,32,16,8,4])
        self.fc_mid = nn.Linear(fc_in_features, 32)
        self.fc_out = nn.Linear(32, num_classes)

    def forward(self, x):
        fpn_out, _ = self.effdet(x)
        fpn_out = list(map(lambda t: torch.flatten(t, start_dim=1), fpn_out))

        fpn_out = torch.cat(fpn_out, dim=1)
        out = self.fc_out(self.fc_mid(fpn_out))
        return out

if __name__ == '__main__':

    x = torch.randn(20, 3, 512, 512)
    model = EfficientClassification(num_classes=2)
    fpn_out = model(x)
    print('FIN')