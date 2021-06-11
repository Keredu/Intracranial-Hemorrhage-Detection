import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve,
                             precision_recall_curve as pr_curve,
                             auc,
                             accuracy_score)

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
    return roc_auc

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
    return pr_auc


def accuracy(ground_truth, inferences, threshold=0.5):
    return accuracy_score(ground_truth, inferences > threshold).item()
