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
from train import train_model
from model import initialize_model
from datasets import IHDataset

dataset_dir = './data/windowed/'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([transforms.Resize((224,224)),
                               transforms.RandomVerticalFlip(p=0.5),
                               transforms.RandomHorizontalFlip(p=0.5),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])
transform_val = transforms.Compose([transforms.Resize((224,224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                               ])
# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
#training_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform) # Data augmentation is only done on training images
#validation_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)

train_dataset = IHDataset(root_dir=dataset_dir, stage='train')
valid_dataset = IHDataset(root_dir=dataset_dir, stage='valid')

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = 'resnet'

# Number of classes in the dataset
num_classes = 2

# Batch size for training (change depending on how much memory you have)
batch_size = 128

# Number of epochs to train for
num_epochs = 20

# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
feature_extract = False

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True , num_workers=8, pin_memory=True)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

# Initialize the model for this run
model, input_size = initialize_model(model_name, num_classes, feature_extract, use_pretrained=True)

# Print the model we just instantiated
#print(model)

dataloaders_dict = {'train':train_loader,'val':valid_loader}
#dataloaders_dict = {'train':valid_loader,'val':valid_loader}

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Send the model to GPU
model = model.to(device)
# Gather the parameters to be optimized/updated in this run. If we are
#  finetuning we will be updating all parameters. However, if we are
#  doing feature extract method, we will only update the parameters
#  that we have just initialized, i.e. the parameters with requires_grad
#  is True.
params_to_update = model.parameters()
print("Params to learn:")
if feature_extract:
    params_to_update = []
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            print("\t",name)
else:
    for name,param in model.named_parameters():
        if param.requires_grad == True:
            print("\t",name)

# Observe that all parameters are being optimized
optimizer = optim.SGD(params_to_update, lr=0.01, momentum=0.9)

# Setup the loss fxn
criterion = nn.CrossEntropyLoss()

# Setup the learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)

# Train and evaluate
model, hist = train_model(model, dataloaders_dict, criterion, optimizer, scheduler, num_epochs)

print('Saving weights...', end='')
torch.save(model.state_dict(), './IH_resnet_weights.pt')
print(' Done.')

