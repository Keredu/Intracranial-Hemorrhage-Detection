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

def train_model(conf):
    device = conf['device']
    model = conf['model'].to(device)
    dataloaders = conf['dataloaders']
    criterion = conf['criterion']
    optimizer = conf['optimizer']
    scheduler = conf['scheduler']
    num_epochs = conf['num_epochs']
    experiment_dir = conf['experiment_dir']

    since = time.time()

    val_acc_history = []

    best_weights = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs} | LR = {scheduler.get_last_lr()}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            n_batches = len(dataloaders[phase])
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
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
                if (i+1)%(n_batches//3) == 0: print(f'Batch {i+1}/{n_batches}')

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_weights = copy.deepcopy(model.state_dict())
                weights_file = f'epoch{epoch}_vacc{best_acc:.3f}_vloss{epoch_loss:.3f}_{model.name}.pt'
                weights_path = os.path.join(experiment_dir, weights_file)
                torch.save(best_weights, weights_path)
            if phase == 'val':
                val_acc_history.append(epoch_acc)
        scheduler.step()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_weights)

    return model, val_acc_history


if __name__ == '__main__':
    from config import get_config

    # Get config from conf.yaml
    conf = get_config('training_conf.yaml')

    # Train and evaluate
    model, hist = train_model(conf)

