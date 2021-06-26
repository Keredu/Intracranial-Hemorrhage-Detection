import sys
sys.path.append('.')
from tqdm import tqdm
import torch
from torch.nn import functional as F
from .metrics import calc_accuracy, roc_auc, pr_auc, calc_conf_mat
import os
import yaml
from .grad_cam import gc


def evaluate(conf):
    device = conf['device']
    dataloader = conf['dataloaders']['test']
    experiment_dir = conf['experiment_dir']
    classes = conf['data']['classes']

    model = conf['model']
    model.load_state_dict(conf['best_weights'])
    model = model.to(device)
    model.eval()

    #grad_cam = conf['grad_cam']
    ground_truth = None
    inferences = None

    batch_bar = tqdm(dataloader, desc='Batch', unit='batches', leave=False)
    for inputs, labels in batch_bar:
        inputs = inputs.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)

        probs = F.softmax(outputs, dim=1)[:, 1]
        probs = probs.cpu()
        if ground_truth is None and inferences is None:
            ground_truth = labels
            inferences = probs
        else:
            ground_truth = torch.cat((ground_truth, labels))
            inferences = torch.cat((inferences, probs))

        # Calculate save metrics

    metrics = {'confusion_matrix': calc_conf_mat(ground_truth=ground_truth,
                                                 inferences=inferences,
                                                 normalize='pred',
                                                 threshold=0.5),
               'accuracy': calc_accuracy(ground_truth=ground_truth,
                                         inferences=inferences,
                                         threshold=0.5),
               'roc_auc': roc_auc(ground_truth=ground_truth,
                                  inferences=inferences,
                                  experiment_dir=experiment_dir),
               'pr_auc': pr_auc(ground_truth=ground_truth,
                                inferences=inferences,
                                experiment_dir=experiment_dir),
                }

    with open(os.path.join(experiment_dir, 'metrics.yaml'), 'w') as fp:
        yaml.dump(metrics, fp)
    '''
    if grad_cam:
        gc(model=model,
           dataloader=dataloader,
           experiment_dir=experiment_dir,
           classes=classes,
           device=device)
    '''

if __name__ == '__main__':
    from config import get_config
    conf = get_config('./conf/evaluation.yaml')
    evaluate(conf)