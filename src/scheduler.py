import torch.optim as optim

def get_scheduler(conf):
    '''Setup the learning rate scheduler'''
    scheduler = conf['scheduler']['name']
    if scheduler == 'StepLR':
        optimizer = conf['optimizer']
        step_size = conf['scheduler']['step_size']
        gamma = conf['scheduler']['gamma']
        scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer,
                                              step_size=step_size,
                                              gamma=gamma)
    else:
        print('Scheduler {scheduler} not suported.')
        exit()
    return scheduler