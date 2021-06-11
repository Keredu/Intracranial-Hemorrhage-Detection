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
    # SGD
    if optimizer == 'SGD':
        lr = conf['optimizer']['lr']
        momentum = conf['optimizer'].get('momentum', 0)
        dampening = conf['optimizer'].get('dampening', 0)
        weight_decay = conf['optimizer'].get('weight_decay', 0)
        nesterov = conf['optimizer'].get('nesterov', False)
        optimizer = optim.SGD(params=params_to_update,
                              lr=lr,
                              dampening=dampening,
                              momentum=momentum,
                              weight_decay=weight_decay,
                              nesterov=nesterov)
    # Adam
    elif optimizer == 'Adam':
        lr = conf['optimizer'].get('lr', 0.001)
        beta0 = conf['optimizer']['beta'].get('beta0', 0.9)
        beta1 = conf['optimizer']['beta'].get('beta1', 0.999)
        eps=conf['optimizer'].get('eps', 1e-8)
        weight_decay = conf['optimizer'].get('weight_decay', 0.0)
        amsgrad = conf['optimizer'].get('amsgrad', False)
        optimizer = optim.Adam(params=params_to_update,
                                lr=lr,
                                betas=(beta0, beta1),
                                eps=eps,
                                weight_decay=weight_decay,
                                amsgrad=amsgrad)
    # AdamW
    elif optimizer == 'AdamW':
        lr = conf['optimizer'].get('lr', 0.001)
        beta0 = conf['optimizer']['beta'].get('beta0', 0.9)
        beta1 = conf['optimizer']['beta'].get('beta1', 0.999)
        eps=conf['optimizer'].get('eps', 1e-8)
        weight_decay = conf['optimizer'].get('weight_decay', 0.01)
        amsgrad = conf['optimizer'].get('amsgrad', False)
        optimizer = optim.AdamW(params=params_to_update,
                                lr=lr,
                                betas=(beta0, beta1),
                                eps=eps,
                                weight_decay=weight_decay,
                                amsgrad=amsgrad)
    else:
        print('Optimizer {optimizer} not supported.')
        exit()
    return optimizer