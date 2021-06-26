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
        self.resnet = models.resnet18(pretrained=True)

        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

        self.deconv0 = nn.ConvTranspose2d(in_channels=64,
                                          out_channels=16,
                                          kernel_size=19,
                                          stride=3,
                                          padding=1,
                                          dilation=2)

        self.deconv1 = nn.ConvTranspose2d(in_channels=64,
                                          out_channels=12,
                                          kernel_size=9,
                                          stride=7,
                                          padding=1,
                                          dilation=1)

        self.deconv2 = nn.ConvTranspose2d(in_channels=64,
                                          out_channels=8,
                                          kernel_size=24,
                                          stride=9,
                                          padding=2,
                                          dilation=4)

        self.deconv3 = nn.ConvTranspose2d(in_channels=64,
                                          out_channels=4,
                                          kernel_size=28,
                                          stride=9,
                                          padding=1,
                                          dilation=6)

        self.deconv4 = nn.ConvTranspose2d(in_channels=64,
                                          out_channels=2,
                                          kernel_size=30,
                                          stride=8,
                                          padding=2,
                                          dilation=7)

        self.conv0 = nn.Conv2d(in_channels=42,
                              out_channels=16,
                              kernel_size=5,
                              padding=2)

        self.conv1 = nn.Conv2d(in_channels=16,
                              out_channels=3,
                              kernel_size=3,
                              padding=1)

    def forward(self, x):
        # EffNet + BiFPN
        fpn_out, _ = self.effdet(x)

        # Convolution Transpose
        out0 = self.deconv0(fpn_out[0])
        out1 = self.deconv1(fpn_out[1])
        out2 = self.deconv2(fpn_out[2])
        out3 = self.deconv3(fpn_out[3])
        out4 = self.deconv4(fpn_out[4])
        deconv_out = torch.cat([out0,out1,out2,out3,out4], dim=1)

        # Convolution
        conv_out = self.conv1(self.conv0(deconv_out))

        # Resnet18
        out = self.resnet(conv_out)
        return out

class EfficientClassification2(nn.Module):

    def __init__(self, num_classes):
        super(EfficientClassification2, self).__init__()
        from effdet import create_model
        self.effdet = create_model(model_name='efficientdet_d0')
        self.effdet.box_net = nn.Identity()
        self.effdet.class_net = nn.Identity()

        # In features from FPN
        fc_in_features = [64 * i*i for i in [64,32,16,8,4]]
        mid = 64
        self.fc0 = nn.Linear(fc_in_features[0], mid)
        self.fc1 = nn.Linear(fc_in_features[1], mid)
        self.fc2 = nn.Linear(fc_in_features[2], mid)
        self.fc3 = nn.Linear(fc_in_features[3], mid)
        self.fc4 = nn.Linear(fc_in_features[4], mid)
        self.fc_out = nn.Linear(5 * mid, num_classes)

    def forward(self, x):
        fpn_out, _ = self.effdet(x)
        fpn_out = list(map(lambda t: torch.flatten(t, start_dim=1), fpn_out))
        out0 = self.fc0(fpn_out[0])
        out1 = self.fc1(fpn_out[1])
        out2 = self.fc2(fpn_out[2])
        out3 = self.fc3(fpn_out[3])
        out4 = self.fc4(fpn_out[4])
        fc_outs = torch.cat([out0,out1,out2,out3,out4], dim=1)
        out = self.fc_out(fc_outs)
        return out

if __name__ == '__main__':

    x = torch.randn(20, 3, 512, 512)
    model = EfficientClassification(num_classes=2)
    fpn_out = model(x)
    print('FIN')