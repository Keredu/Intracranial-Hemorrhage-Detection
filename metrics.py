import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score
import os
import numpy as np

def roc_auc(ground_truth, inferences, experiment_dir=None):
    # calculate the fpr and tpr for all thresholds of the classification
    fpr, tpr, threshold = roc_curve(ground_truth, inferences)
    roc_auc = auc(fpr, tpr).item()
    if not experiment_dir is None:
        plt.title('Receiver Operating Characteristic')
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.3f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.savefig(os.path.join(experiment_dir, 'roc_auc.png'))
    return roc_auc

def accuracy(ground_truth, inferences, threshold=0.5):
    return accuracy_score(ground_truth, inferences > threshold).item()
