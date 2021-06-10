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
from tqdm import tqdm

def train_model(conf):
    device = conf['device']
    model = conf['model'].to(device)
    dataloaders = conf['dataloaders']
    criterion = conf['criterion']
    optimizer = conf['optimizer']
    scheduler = conf['scheduler']
    num_epochs = conf['num_epochs']
    experiment_dir = conf['experiment_dir']

    valid_acc_history = []
    best_acc = 0.0

    epoch_bar = tqdm(range(num_epochs), desc='Epoch',unit="epochs")
    for epoch in epoch_bar:
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            n_batches = len(dataloaders[phase])

            batch_bar = tqdm(dataloaders[phase], desc='Batch',unit="batches", leave=False)
            for inputs, labels in batch_bar:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                batch_bar.set_postfix(phase=phase, batch_loss=loss.item())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)


            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_weights = copy.deepcopy(model.state_dict())
                weights_file = f'epoch{epoch}_vacc{best_acc:.3f}_vloss{epoch_loss:.3f}_{model.name}.pt'
                weights_path = os.path.join(experiment_dir, weights_file)
                torch.save(best_weights, weights_path)
            if phase == 'valid':
                valid_acc_history.append(epoch_acc)

        epoch_bar.set_postfix(vloss=epoch_loss, vacc=epoch_acc.item())
        scheduler.step()

    print('Best valid Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_weights)

    return model, valid_acc_history


if __name__ == '__main__':
    from config import get_config

    # Get config from conf.yaml
    conf = get_config('training_conf.yaml')

    # Train and evaluate
    model, hist = train_model(conf)

