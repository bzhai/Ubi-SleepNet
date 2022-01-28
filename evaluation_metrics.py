import pandas as pd
import numpy as np
from imblearn.metrics import specificity_score
from sklearn import metrics
from joblib import Parallel, delayed
import multiprocessing

def eval_precision(gt, pred, average='macro'):
    if type(gt) is pd.core.frame.DataFrame or type(gt) is pd.core.frame.Series:
        return metrics.precision_score(gt.fillna(0.0), pred.fillna(0.0), average=average)*100.0
    else:
        return metrics.precision_score(gt, pred, average=average)*100.0


def eval_acc(gt, pred, average='macro'):
    if type(gt) is pd.core.frame.DataFrame or type(gt) is pd.core.frame.Series:
        return metrics.accuracy_score(gt.fillna(0.0), pred.fillna(0.0))*100.0
    else:
        return metrics.accuracy_score(gt, pred)*100.0

def eval_cohen(gt, pred, average='quadratic'):

    if average != 'binary':
        weight_method = "quadratic"
    else:
        weight_method = None
    if type(gt) is pd.core.frame.DataFrame or type(gt) is pd.core.frame.Series:
        return metrics.cohen_kappa_score(gt.fillna(0.0), pred.fillna(0.0), weights=weight_method)*100.0
    else:
        return metrics.cohen_kappa_score(gt, pred, weights=weight_method)*100.0

def eval_acc_multiple_classes(gt, pred, label=0):
    tmp_pd = pd.DataFrame()
    tmp_pd['gt'] = gt
    tmp_pd['pred'] = pred
    tmp_pd = tmp_pd[(tmp_pd['gt'] == label) | (tmp_pd['pred'] == label)]
    tmp_pd['pred'] = tmp_pd['pred'].apply(lambda x: 1 if x == label else 0)
    tmp_pd['gt'] = tmp_pd['gt'].apply(lambda x: 1 if x == label else 0)
    return metrics.accuracy_score(tmp_pd['gt'], tmp_pd['pred'])

def eval_recall(gt, pred, average='macro'):
    if type(gt) is pd.core.frame.DataFrame or type(gt) is pd.core.frame.Series:
        return metrics.recall_score(gt.fillna(0.0), pred.fillna(0.0), average=average)* 100.0
    else:
        return metrics.recall_score(gt, pred, average=average) * 100.0


def eval_specificity(gt, pred, average='macro'):
    if type(gt) is pd.core.frame.DataFrame or type(gt) is pd.core.frame.Series:
        return specificity_score(gt.fillna(0.0), pred.fillna(0.0), average=average) * 100.0
    else:
        return specificity_score(gt, pred, average=average) * 100.0

def eval_f1(gt, pred, average='macro'):
    if type(gt) is pd.core.frame.DataFrame or type(gt) is pd.core.frame.Series:
        return metrics.f1_score(gt.fillna(0.0), pred.fillna(0.0), average=average) * 100.0
    else:
        return metrics.f1_score(gt, pred, average='macro') * 100.0


def applyParallel(dfGrouped, func):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count())(delayed(func)(group) for name, group in dfGrouped)
    return pd.concat(retLst)

def applyParallel_1(dfGrouped, index, func, alg1, alg2):
    retLst = Parallel(n_jobs=multiprocessing.cpu_count()/2)(delayed(func)(group, alg1, alg2) for name, group in dfGrouped)
    t1 = pd.Series(retLst, index)
    return t1

def eval_acc_df(df,alg1,alg2, average='binary'):
    return metrics.accuracy_score(df[alg1].fillna(0.0), df[alg2].fillna(0.0))

def calc_metrics(y_gt, y_pred_all, avg_method = "macro"):
    """
    average method should be "binary" or "macro"
    """
    accuracy = eval_acc(y_gt, y_pred_all)
    macro_f1 = eval_f1(y_gt, y_pred_all, average=avg_method)
    specificity = eval_specificity(y_gt, y_pred_all, average=avg_method)
    precision = eval_precision(y_gt, y_pred_all, average=avg_method)
    cohen = eval_cohen(y_gt, y_pred_all, average=avg_method)
    recall = eval_recall(y_gt, y_pred_all, average=avg_method)
    return {avg_method + '_accuracy': accuracy, avg_method+'_specificity': specificity,
            avg_method+'_precision': precision, avg_method+'_f1': macro_f1,
            avg_method+'_cohen': cohen, avg_method+'_recall': recall
            }
