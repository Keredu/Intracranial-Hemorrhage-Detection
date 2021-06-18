import torch
from torch.nn import functional as F
import os
import yaml
from grad_cam_test import gc_test


def test(conf):
    device = conf['device']
    dataset = conf['test_dataset']
    classes = conf['data']['classes']
    weights_path = conf['weights_path']
    results_dir = conf['results_dir']

    model = conf['model']
    model.load_state_dict(torch.load(weights_path))
    model = model.to(device)
    model.eval()

    gc_test(model=model,
       dataset=dataset,
       experiment_dir=results_dir,
       classes=classes,
       device=device)


if __name__ == '__main__':
    from config import get_config
    conf = get_config('./conf/testing.yaml')
    test(conf)