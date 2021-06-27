import sys
sys.path.append('.')
from tqdm import tqdm
import torch
from torch.nn import functional as F
from .metrics import roc_auc, pr_auc, calc_metrics
import os
import yaml
import numpy as np


def evaluate(conf):
    device = conf['device']
    dataloader = conf['dataloaders']['test']
    experiment_dir = conf['experiment_dir']
    classes = conf['data']['classes']
    batch_size = conf['data']['batch_size']

    model = conf['model']
    model.load_state_dict(conf['best_weights'])
    model = model.to(device)
    model.eval()


    ground_truth = None
    inferences = None

    batch_bar = tqdm(dataloader, desc='Batch', unit='batches', leave=True)
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

    metrics = {'metrics0.5': calc_metrics(ground_truth=ground_truth,
                                          inferences=inferences,
                                          threshold=0.5),
               'metrics0.7': calc_metrics(ground_truth=ground_truth,
                                          inferences=inferences,
                                          threshold=0.7),
               'metrics0.9': calc_metrics(ground_truth=ground_truth,
                                          inferences=inferences,
                                          threshold=0.9),
               'roc_auc': roc_auc(ground_truth=ground_truth,
                                  inferences=inferences,
                                  experiment_dir=experiment_dir),
               'pr_auc': pr_auc(ground_truth=ground_truth,
                                inferences=inferences,
                                experiment_dir=experiment_dir),
                }

    patients_dataset = conf['patients_dataset']
    test_patients = patients_dataset.test_patients
    patients_bar = tqdm(test_patients.items(),
                        desc='Patient', unit='patients', leave=True)

    inferences = {1: [], 2:[], 3: [], 4: [], 5: []}
    ground_truth = []
    for patient, patient_data in patients_bar:
        patient_IH = patient_data['IH'] # If the patient has IH or not
        slices = patient_data['slices_IH'] + patient_data['slices_noIH']
        slices_with_IH = 0
        samples = []
        for slice_id in slices:
            sample, _ = patients_dataset.getSlice(slice_id)
            samples.append(sample)

        for i in range(0, len(samples), batch_size):
            batch = torch.stack(samples[i:i+batch_size], dim=0)
            batch = batch.to(device)
            with torch.set_grad_enabled(False):
                outputs = model(batch)
            IH_probs = F.softmax(outputs, dim=1)[:, 1]
            slices_with_IH += (IH_probs > 0.8).sum()
        for num_IH_threshold in [1,2,3,4,5]:
            net_IH_prediction = slices_with_IH >= num_IH_threshold
            inferences[num_IH_threshold].append(net_IH_prediction)
        ground_truth.append(patient_IH)

    ground_truth = np.array(ground_truth).astype(float)
    inferences1 = np.array(inferences[1]).astype(float)
    inferences2 = np.array(inferences[2]).astype(float)
    inferences3 = np.array(inferences[3]).astype(float)
    inferences4 = np.array(inferences[4]).astype(float)
    inferences5 = np.array(inferences[5]).astype(float)
    metrics['patients_metrics (>= 1 IH slice)'] = calc_metrics(
                                                    ground_truth=ground_truth,
                                                    inferences=inferences1)
    metrics['patients_metrics (>= 2 IH slice)'] = calc_metrics(
                                                    ground_truth=ground_truth,
                                                    inferences=inferences2)
    metrics['patients_metrics (>= 3 IH slice)'] = calc_metrics(
                                                    ground_truth=ground_truth,
                                                    inferences=inferences3)
    metrics['patients_metrics (>= 4 IH slice)'] = calc_metrics(
                                                    ground_truth=ground_truth,
                                                    inferences=inferences4)
    metrics['patients_metrics (>= 5 IH slice)'] = calc_metrics(
                                                    ground_truth=ground_truth,
                                                    inferences=inferences5)

    del metrics['patients_metrics (>= 1 IH slice)']['threshold']
    del metrics['patients_metrics (>= 2 IH slice)']['threshold']
    del metrics['patients_metrics (>= 3 IH slice)']['threshold']
    del metrics['patients_metrics (>= 4 IH slice)']['threshold']
    del metrics['patients_metrics (>= 5 IH slice)']['threshold']

    with open(os.path.join(experiment_dir, 'metrics.yaml'), 'w') as fp:
        yaml.dump(metrics, fp)