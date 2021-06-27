import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (precision_recall_curve as pr_curve,
                             confusion_matrix,
                             precision_score,
                             accuracy_score,
                             recall_score,
                             roc_curve,
                             auc)

def roc_auc(ground_truth, inferences, experiment_dir):
    fpr, tpr, threshold = roc_curve(ground_truth, inferences)
    roc_auc = auc(fpr, tpr).item()
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig(os.path.join(experiment_dir, 'roc_auc.png'))
    plt.clf()
    return round(roc_auc, 3)


def pr_auc(ground_truth, inferences, experiment_dir):
    precision, recall, threshold = pr_curve(ground_truth, inferences)
    #precision, recall, threshold = sorted(zip(precision, recall))
    pr_auc = auc(recall, precision).item()
    plt.title('Precision - Recall')
    plt.plot(recall, precision, 'b', label = 'AUC = %0.3f' % pr_auc)
    plt.legend(loc = 'lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.savefig(os.path.join(experiment_dir, 'pr_auc.png'))
    plt.clf()
    return round(pr_auc, 3)


def calc_metrics(ground_truth, inferences, normalize=None, threshold=0.5):
    y_pred = inferences > threshold
    accuracy = accuracy_score(y_true=ground_truth, y_pred=y_pred).item()
    recall = recall_score(y_true=ground_truth, y_pred=y_pred).item()
    precision = precision_score(y_true=ground_truth, y_pred=y_pred).item()

    conf_mat = confusion_matrix(y_true=ground_truth,
                                y_pred=y_pred,
                                normalize=normalize)
    tn, fp, fn, tp = list(map(lambda x: x.item(), conf_mat.ravel()))
    metrics = {'threshold': threshold, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
               'precision': precision, 'recall': recall, 'accuracy': accuracy}
    return {k: round(v, 3) for k,v in metrics.items()}