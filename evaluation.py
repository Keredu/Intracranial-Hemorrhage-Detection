from tqdm import tqdm
import torch
from torch.nn import functional as F
from metrics import roc_auc, accuracy
import os
import yaml

def evaluate(conf):
    device = conf['device']
    dataloader = conf['dataloaders']['valid']
    experiment_dir = conf['experiment_dir']

    model = conf['model']
    model.load_state_dict(conf['best_weights'])
    model = model.to(device)
    model.eval()

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

    metrics = {'accuracy': accuracy(ground_truth=ground_truth,
                                    inferences=inferences),
               'roc_auc': roc_auc(ground_truth=ground_truth,
                                  inferences=inferences,
                                  experiment_dir=experiment_dir)
              }

    with open(os.path.join(experiment_dir, 'metrics.yaml'), 'w') as fp:
        yaml.dump(metrics, fp)


if __name__ == '__main__':
    from config import get_config
    conf = get_config('./conf/evaluation.yaml')
    evaluate(conf)