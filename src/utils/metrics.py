import numpy as np
from sklearn.metrics import roc_auc_score


def image_auroc(labels, scores):
    return roc_auc_score(labels, scores)


def pixel_auroc(masks, maps):
    return roc_auc_score(np.asarray(masks).ravel() > 0.5, np.asarray(maps).ravel())