import torch.optim as optim

def get_params_to_update(model, print_params=True):
    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = []
    if print_params: print("Params to learn:")
    for name, param in model.named_parameters():
        if param.requires_grad == True:
            params_to_update.append(param)
            if print_params: print(name)
    return params_to_update

def get_optimizer(conf):
    model = conf['model']
    print_params=conf['optimizer']['print_params']
    params_to_update = get_params_to_update(model=model,
                                            print_params=print_params)

    optimizer = conf['optimizer']['name']
    if optimizer == 'SGD':
        lr = conf['optimizer']['lr']
        momentum = conf['optimizer']['momentum']
        weight_decay = conf['optimizer']['weight_decay']
        optimizer = optim.SGD(params=params_to_update,
                              lr=lr,
                              momentum=momentum,
                              weight_decay=weight_decay)
    else:
        print('Optimizer {optimizer} not supported.')
        exit()
    return optimizer