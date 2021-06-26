import torch
import os
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt

def save_losses_graph(train_losses, valid_losses, experiment_dir):
    epochs = range(len(train_losses))
    plt.plot(epochs, train_losses, 'g', label='Training loss')
    plt.plot(epochs, valid_losses, 'b', label='Validation loss')
    plt.title('Training and Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc = 'upper right')
    plt.savefig(os.path.join(experiment_dir, 'train_valid_loss.png'))
    plt.clf()

def save_acc_graph(train_accs, valid_accs, experiment_dir):
    epochs = range(len(train_accs))
    plt.plot(epochs, train_accs, 'g', label='Training acc')
    plt.plot(epochs, valid_accs, 'b', label='Validation acc')
    plt.title('Training and Validation acc')
    plt.xlabel('Epochs')
    plt.ylabel('Acc')
    plt.legend(loc = 'upper right')
    plt.savefig(os.path.join(experiment_dir, 'train_valid_acc.png'))
    plt.clf()


def train(conf):
    device = conf['device']
    model = conf['model'].to(device)
    dataloaders = conf['dataloaders']
    criterion = conf['criterion']
    optimizer = conf['optimizer']
    scheduler = conf['scheduler']
    num_epochs = conf['num_epochs']
    experiment_dir = conf['experiment_dir']

    #valid_acc_history = []
    best_acc = 0.0

    epoch_bar = tqdm(range(num_epochs), desc='Epoch',unit='epoch')
    train_losses = []
    valid_losses = []
    train_accs = []
    valid_accs = []
    train_loss = -1.0
    valid_loss = -1.0
    train_acc = -1.0
    valid_acc = -1.0
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
            batch_bar = tqdm(dataloaders[phase],
                             desc='Batch',
                             unit='batch',
                             leave=False)
            batch_losses = []
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
                running_corrects += torch.sum(preds == labels.data)
                batch_loss = loss.item()
                running_loss += batch_loss
                batch_losses.append(batch_loss)


                batch_bar.set_postfix(phase=phase, batch_loss=batch_loss)

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            if phase == 'train':
                train_loss = epoch_loss
                train_acc = epoch_acc.item()
                train_losses.append(train_loss)
                train_accs.append(train_acc)
            else:
                valid_loss = epoch_loss
                valid_acc = epoch_acc
                valid_losses.append(valid_loss)
                valid_accs.append(valid_acc)
                save_losses_graph(train_losses, valid_losses, experiment_dir)
                save_acc_graph(train_accs, valid_accs, experiment_dir)

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_weights = copy.deepcopy(model.state_dict())
                weights_file = 'best_weights.pt'
                weights_path = os.path.join(experiment_dir, weights_file)
                torch.save(best_weights, weights_path)

        epoch_bar.set_postfix(tloss=train_loss,tacc=train_acc,
                              vloss=valid_loss, vacc=valid_acc)
        scheduler.step()

    print('Best valid Acc: {:4f}'.format(best_acc))

if __name__ == '__main__':
    from config import get_config
    # Get config from conf.yaml
    conf = get_config('./conf/training.yaml')

    # Train model
    train(conf)
