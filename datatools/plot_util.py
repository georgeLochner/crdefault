import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, average_precision_score


import matplotlib.pyplot as plt
import seaborn as sns

def display_importances(feature_importance_df_,imagepath="imp.png"):
    # Plot feature importances
    feature_importance_df_['mean']=feature_importance_df_.mean(axis=1)
    feature_importance_df_= feature_importance_df_.sort_values(by='mean',ascending=False)[:50]

    impdata=pd.DataFrame()
    for fold in feature_importance_df_.columns.values:
        df=pd.DataFrame(index=feature_importance_df_.index)
        df['feat']=feature_importance_df_.index.values
        df['imp']=feature_importance_df_[fold]
        df['fold']=fold
        impdata=pd.concat([impdata, df],axis=0)

    plt.figure(figsize=(8,10))
    sns.barplot(data=impdata,y='feat',x='imp')
    plt.title('LightGBM Features (avg over folds)')
    plt.tight_layout()
    plt.savefig(imagepath)


def display_roc_curve(y_, oof_preds_, folds_idx_,maxfold=-1,imagepath="roc.png"):
    # Plot ROC curves
    plt.figure(figsize=(6,6))
    scores = []
    for n_fold, (_, val_idx) in enumerate(folds_idx_):
        if n_fold<maxfold:
            # Plot the roc curve
            fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
            score = roc_auc_score(y_.iloc[val_idx], oof_preds_[val_idx])
            scores.append(score)
            plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (n_fold + 1, score))
    y_=y_[~(np.isnan(oof_preds_))]
    oof_preds_=oof_preds_[~(np.isnan(oof_preds_))]
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    fpr, tpr, thresholds = roc_curve(y_, oof_preds_)
    score = roc_auc_score(y_, oof_preds_)
    plt.plot(fpr, tpr, color='b',
             label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
             lw=2, alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('LightGBM ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()

    plt.savefig(imagepath)


def display_precision_recall(y_, oof_preds_, folds_idx_,maxfold=-1,imagepath="precall.png"):
    # Plot ROC curves
    plt.figure(figsize=(6,6))

    scores = []
    if folds_idx_ is not None:
        for n_fold, (_, val_idx) in enumerate(folds_idx_):
            if n_fold<maxfold:
                # Plot the roc curve
                fpr, tpr, thresholds = roc_curve(y_.iloc[val_idx], oof_preds_[val_idx])
                score = average_precision_score(y_.iloc[val_idx], oof_preds_[val_idx])
                scores.append(score)
                plt.plot(fpr, tpr, lw=1, alpha=0.3, label='AP fold %d (AUC = %0.4f)' % (n_fold + 1, score))
    y_=y_[~(np.isnan(oof_preds_))]
    oof_preds_=oof_preds_[~(np.isnan(oof_preds_))]
    precision, recall, thresholds = precision_recall_curve(y_, oof_preds_)
    score = average_precision_score(y_, oof_preds_)
    plt.plot(precision, recall, color='b',
             label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
             lw=2, alpha=.8)

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('LightGBM Recall / Precision')
    plt.legend(loc="best")
    plt.tight_layout()

    plt.savefig(imagepath)


