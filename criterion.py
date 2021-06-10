import torch.nn as nn

def get_criterion(conf):
    criterion = conf['criterion']['name']
    if criterion == 'CrossEntropyLoss':
        criterion = nn.CrossEntropyLoss()
    else:
        print(f'Criterion {criterion} not supported.')
        exit()
    return criterion
