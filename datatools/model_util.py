from sklearn.model_selection import StratifiedKFold
import datetime
import pandas as pd
from sklearn.metrics import roc_auc_score
import numpy as np

import gc
import datatools.plot_util as pltutil
import matplotlib.pyplot as plt
import os
import json
import lightgbm as lgb


def train_lgb_cv(data_:pd.DataFrame, y_:pd.Series, params={}, folds_=None, outputPath="./output/", test_=None, maxfold=-1, objective='binary', metrics='auc', num_boost_round=4000, early_stopping_rounds=20, feval=None, saveData=False):
    if folds_ is None: folds_ = StratifiedKFold(n_splits=5, shuffle=True, random_state=546789)
    resultpath=outputPath+datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    if saveData:
        if not os.path.exists(resultpath): os.makedirs(resultpath)
    lgb_params = {
        'boosting_type': 'gbdt',
        'objective': objective,
        'metric':metrics,
        'learning_rate': 0.1, #learning rate: keep it only for training speed, not to tune (otherwise overfitting)
        'num_leaves': 7,  # [7, 4095] we should let it be smaller than 2^(max_depth)
        #ps dOCS SUGGEST 70-80
        'max_depth': 3,  #[2, 63] -1 means no limit
        #'min_data_in_leaf': 200, #Nb param
        #Speed + Overfitting
        'colsample_bytree': 0.9,  #1.0 [0.4, 1] Subsample ratio of columns when constructing each tree.
        #           Two of the most important for this competition are feature_fraction and reg_lambda.
        # 'subsample': 0.9,  #[0.4, 1] bagging_fraction Subsample ratio of the training instance.
        # 'subsample_freq': 0,  #bagging_freq frequence of subsample, <=0 means no enable
        #Overfitting
        # 'reg_alpha': .1,  #0.0 L1 regularization term on weights
        #              Two of the most important for this competition are feature_fraction and reg_lambda.
        # 'reg_lambda': .1,  #0.0 L2 regularization term on weights
        # 'min_split_gain': .01,  #0.0 lambda_l1, lambda_l2 and min_gain_to_split to regularization
        # 'min_child_weight': 2,  #1e-3  [0.01, (sample size / 1000)] Minimum sum of instance weight(hessian) needed in a child(leaf)
        # 'silent': -1,
        'verbose': -1
        # 'max_bin': 100,  #255 use small if overfit Number of bucketed bin for feature values
        #       max_bin: unbalanced - keep it only for memory pressure, not to tune (otherwise overfitting)
        # 'subsample_for_bin': 200000,  # 200000 set larger for sparse data - Number of samples for constructing bin
        # 'nthread': 4
    }

    lgb_params.update(params)
    printprm=lgb_params.copy()

    oof_preds = np.full(data_.shape[0],np.nan)

    if test_ is not None:
        sub_preds = np.zeros(test_.shape[0])

    feature_importance_df = pd.DataFrame()
    feats = [f for f in data_.columns if f not in ['SK_ID_CURR']]
    if maxfold < 0: maxfold = folds_.n_splits
    auc_fold=np.array([])
    for n_fold, (trn_idx, val_idx) in enumerate(folds_.split(data_,y_)):
        if n_fold == maxfold: break
        trn_x, trn_y = data_[feats].iloc[trn_idx], y_.iloc[trn_idx]
        xgtrain = lgb.Dataset(trn_x, label=trn_y)
        val_x, val_y = data_[feats].iloc[val_idx], y_.iloc[val_idx]
        xgvalid = lgb.Dataset(val_x, label=val_y)

        evals_results = {}
        clf = lgb.train(lgb_params,
                        xgtrain,
                        valid_sets=[xgtrain, xgvalid],
                        valid_names=['train','valid'],
                        evals_result=evals_results,
                        num_boost_round=num_boost_round,
                        early_stopping_rounds=early_stopping_rounds,
                        verbose_eval=10,
                        feval=feval)
        if saveData:
            ax = lgb.plot_metric(evals_results, metric='auc')
            plt.savefig(resultpath+"/lrate"+str(n_fold)+".png")

        oof_preds[val_idx] = clf.predict(val_x, num_iteration=clf.best_iteration)
        if test_ is not None:
            sub_preds += clf.predict(test_[feats], num_iteration=clf.best_iteration) / folds_.n_splits

        fold_imp = pd.Series(clf.feature_importance(),index=clf.feature_name())

        feature_importance_df[str(n_fold + 1)] = fold_imp
        aucresult=roc_auc_score(val_y, oof_preds[val_idx])
        print('Fold %2d AUC : %.6f' % (n_fold + 1,aucresult ))
        auc_fold =np.append(auc_fold ,aucresult)
        del clf, trn_x, trn_y, val_x, val_y
        gc.collect()
    y_roc=y_[~(np.isnan(oof_preds))]
    oof_preds_=oof_preds[~(np.isnan(oof_preds))]
    cvroc=roc_auc_score(y_roc, oof_preds_)
    print('Full AUC score %.6f' % cvroc)
    if saveData:
        printprm.update({'auc':cvroc})
        with open(resultpath+'/params.json', 'w') as fp:
            json.dump(printprm, fp,indent=1)
        folds_idx = [(trn_idx, val_idx) for trn_idx, val_idx in folds_.split(data_,y_)]
        pltutil.display_importances(feature_importance_df_=feature_importance_df,imagepath=resultpath+"/imp.png")
        pltutil.display_roc_curve(y_=y_, oof_preds_=oof_preds, folds_idx_=folds_idx,maxfold=maxfold,imagepath=resultpath+"/roc.png")
        pltutil.display_precision_recall(y_=y_, oof_preds_=oof_preds, folds_idx_=folds_idx,maxfold=maxfold,imagepath=resultpath+"/prcall.png")
        feature_importance_df.to_pickle(resultpath+"/importance.pkl")
    if test_ is not None:
        test_['TARGET'] = sub_preds
    result ={'oof_preds':oof_preds, 'test':test_, 'feature_importance':feature_importance_df, 'auc-fold':pd.Series(auc_fold),'auc-mean':cvroc}
    return result


defaultLGBparams = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric':'auc',
    'learning_rate': 0.1, #learning rate: keep it only for training speed, not to tune (otherwise overfitting)
    'num_leaves': 7,  # [7, 4095] we should let it be smaller than 2^(max_depth)
    #ps dOCS SUGGEST 70-80
    'max_depth': 3,  #[2, 63] -1 means no limit
    #'min_data_in_leaf': 200, #Nb param
    #Speed + Overfitting
    'colsample_bytree': 0.9,  #1.0 [0.4, 1] Subsample ratio of columns when constructing each tree.
    #           Two of the most important for this competition are feature_fraction and reg_lambda.
    # 'subsample': 0.9,  #[0.4, 1] bagging_fraction Subsample ratio of the training instance.
    # 'subsample_freq': 0,  #bagging_freq frequence of subsample, <=0 means no enable
    #Overfitting
    # 'reg_alpha': .1,  #0.0 L1 regularization term on weights
    #              Two of the most important for this competition are feature_fraction and reg_lambda.
    # 'reg_lambda': .1,  #0.0 L2 regularization term on weights
    # 'min_split_gain': .01,  #0.0 lambda_l1, lambda_l2 and min_gain_to_split to regularization
    # 'min_child_weight': 2,  #1e-3  [0.01, (sample size / 1000)] Minimum sum of instance weight(hessian) needed in a child(leaf)
    # 'silent': -1,
    'verbose': -1
    # 'max_bin': 100,  #255 use small if overfit Number of bucketed bin for feature values
    #       max_bin: unbalanced - keep it only for memory pressure, not to tune (otherwise overfitting)
    # 'subsample_for_bin': 200000,  # 200000 set larger for sparse data - Number of samples for constructing bin
    # 'nthread': 4
}

def scoreLGBModel(train,target,params,seed=0,num_boost=5000,early_stop=20):
    """Scores an LightGBM model"""
    booster=None
    def lgbCallback(lgbResult):
        nonlocal booster
        booster=lgbResult.model
    trainDS = lgb.Dataset(data = train, label = target)
    cv_results = lgb.cv(
        params,
        trainDS,
        num_boost_round=num_boost,
        nfold=5,
        stratified=True,
        early_stopping_rounds=early_stop,
        verbose_eval=20,
        seed=seed,
        callbacks=[lgbCallback]
    )
    foldauc=pd.Series([x[0][2] for x in booster.eval_valid()])
    result ={'auc-fold':foldauc,'auc-mean':foldauc.mean(),'auc-std':foldauc.std()}
    return result, booster

def evalLGBModelAUC(x,y,evalx,evaly,params,seed=0,num_boost=100,early_stop=20):
    """Trains an LGB model against an eval set, returning auc"""
    xgtrain = lgb.Dataset(x, label=y)
    #xgvalid = lgb.Dataset(evalx, label=evaly)
    raise Exception("Not tested yet-create_valid used to create validation set")
    xgvalid = xgtrain.create_valid(evalx,label=evaly)
    clf = lgb.train(params,
                    xgtrain,
                    valid_sets=[xgtrain, xgvalid],
                    valid_names=['train','valid'],
                    num_boost_round=num_boost,
                    early_stopping_rounds=early_stop,
                    seed=seed,
                    verbose_eval=100)
    pred = clf.predict(evalx, num_iteration=clf.best_iteration)
    aucresult=roc_auc_score(evaly, pred)
    return aucresult, clf