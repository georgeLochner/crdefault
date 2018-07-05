from random import shuffle

from scipy.stats import stats


import datatools.datastore_util as du
import pandas as pd
import numpy as np


import datatools.model_util as models


def processFoldResult(baselineResult,result):
    baselineScore = baselineResult['auc-mean']
    baselineFold = baselineResult['auc-fold']
    foldAuc=baselineFold-result['auc-fold']
    tstat=stats.ttest_rel(baselineFold,result['auc-fold'])
    return (result['auc-mean'],result['auc-fold'].mean(),result['auc-fold'].std(),baselineScore-result['auc-mean'],foldAuc.mean(),foldAuc.std(),tstat[1])


def rankFeatureLeaveOneOut(df:pd.DataFrame,target:pd.Series,params,scoringFunc):
    importance=pd.DataFrame(columns=["SCORE","FOLDSCORE","FOLDSCORESTD","GAIN","FOLDGAINMEAN","FOLDGAINSTD","PVAL"])
    cols = np.array(list(df))
    baselineResult=scoringFunc(df,target,params)
    baselineScore = baselineResult['auc-mean']
    baselineFold = baselineResult['auc-fold']
    importance.loc["BASELINE"]=(baselineScore,baselineFold.mean(),baselineFold.std(),0,0,0,1)
    for col in cols:
    #for col in ["DAYS_BIRTH","CODE_GENDER","DAYS_ID_PUBLISH"]:
        print("------------"+col+"-------------------")
        looDF=df.drop(col,axis=1)
        result = scoringFunc(looDF,target,params)
        if 'auc-fold' in result:
            importance.loc[col]=processFoldResult(baselineResult,result)
        else:
            raise Exception("Need to implement - store and sort by gain")
        print("score :    "+str(result['auc-mean']))
        print("baseline : "+str(baselineScore))
        print("gain :     "+str(baselineScore-result['auc-mean']))
    importance=importance.sort_values("PVAL",ascending=True)
    return importance

#coreFeatures - features to be included in every test. Remaining features are tested on at a time, with core features
def rankFeatureAddOneBack(df:pd.DataFrame,target:pd.Series,coreFeatures,testFeatures,params,scoringFunc)->pd.DataFrame:
    importance=pd.DataFrame(columns=["SCORE","FOLDSCORE","FOLDSCORESTD","GAIN","FOLDGAINMEAN","FOLDGAINSTD","PVAL"])
    coredf=df[coreFeatures]
    baselineResult=scoringFunc(coredf,target,params)
    baselineScore = baselineResult['auc-mean']
    baselineFold = baselineResult['auc-fold']
    importance.loc["BASELINE"]=(baselineScore,baselineFold.mean(),baselineFold.std(),0,0,0,1)
    for col in testFeatures:
        print("------------------"+col+"------------------")
        testdf=df[coreFeatures]
        testdf[col]=df[col]
        result = scoringFunc(testdf,target,params)
        if 'auc-fold' in result:
            importance.loc[col]=processFoldResult(baselineResult,result)
        else:
            raise Exception("Need to implement - store and sort by gain")
        print(col+" : ")
        print("score :    "+str(result['auc-mean']))
        print("baseline : "+str(baselineScore))
        print("gain :     "+str(baselineScore-result['auc-mean']))
    importance=importance.sort_values("PVAL",ascending=True)
    importance["GAIN"]=-importance["GAIN"]
    importance["FOLDGAINMEAN"]=-importance["FOLDGAINMEAN"]

    importanceFinal=pd.DataFrame(columns=["SCORE","FOLDSCORE","FOLDSCORESTD","GAIN","FOLDGAINMEAN","FOLDGAINSTD","PVAL","ORDER"])
    order=int(1)
    for colname, row in importance.iterrows():
        if colname=="BASELINE": continue
        coredf[colname]=df[colname]

        result = scoringFunc(coredf,target,params)
        if 'auc-fold' in result:
            importanceFinal.loc[colname]=processFoldResult(baselineResult,result) +(order,)
            importanceFinal["GAIN"][colname]=-importanceFinal["GAIN"][colname]
            importanceFinal["FOLDGAINMEAN"][colname]=-importanceFinal["FOLDGAINMEAN"][colname]
        else:
            raise Exception("Need to implement - store and sort by gain")


        if (importanceFinal["GAIN"][colname] < 0):
            coredf=coredf.drop(colname,axis=1)
        else:
            baselineResult=result
        order+=1
    return importance, importanceFinal


def findUsefulFeatures(train:pd.DataFrame,target:pd.Series,params:dict,scoringFunc,newFeatures=[],coreThreshold=0.001,coreMax=15):
    origFeats=[f for f in train if f not in newFeatures]
    trainOrig=train[origFeats]
    ranking = rankFeatureLeaveOneOut(trainOrig,target,params,scoringFunc)
    print(ranking)
    du.dfToPickle("ranking",ranking)

    if (ranking.iloc[coreMax]["GAIN"]<coreThreshold):
        coreMax = len(ranking[ranking["GAIN"]>=coreThreshold])
    core = list(ranking.index)[:coreMax]
    feats=list(ranking.index)[coreMax:-1] #Last one is the baseline

    # core=['AMT_ANNUITY', 'AMT_GOODS_PRICE', 'DAYS_BIRTH']
    # feats=["CODE_GENDER","DAYS_ID_PUBLISH"]
    ranking2,rankFinal=rankFeatureAddOneBack(trainOrig,target,core,feats,params,scoringFunc)
    print(ranking2)
    print("---------------------------")
    print(rankFinal)
    du.dfToPickle("ranking2",ranking2)
    du.dfToPickle("rankingFinal",rankFinal)


    ranking3,rankFinal3=rankFeatureAddOneBack(train,target,origFeats,newFeatures,params,scoringFunc)
    print(ranking3)
    print("---------------------------")
    print(rankFinal3)
    du.dfToPickle("ranking3",ranking3)
    du.dfToPickle("rankingFinal3",rankFinal3)


#Create a random feature set by select a random initial feature, and adding additional randomly selected features
#which improve the model score
def createRandomFeatureSet(train:pd.DataFrame,target:pd.Series,params:dict,scoringFunc,seed=0,threshold=0.1):
    features=list(train.columns.values)
    shuffle(features)
    coredf=pd.DataFrame()
    results=[]
    baseline=0.5
    for col in features[:]:
        print("------------------"+col+"------------------")
        coredf[col]=train[col]
        result = scoringFunc(coredf,target,params,seed)
        fold=result['auc-fold']
        score=fold.mean()
        stdev=fold.std()
        foldmn = (fold-baseline).mean()
        foldstd = (fold-baseline).std()
        incl = foldmn/foldstd > threshold
        results.append((col,score,stdev,foldmn,foldstd,foldmn/foldstd,incl))
        baseline=fold
        if not incl:
            coredf=coredf.drop(col,axis=1)
            print("-----EXCLUDE------"+col+"------------------")
        print(col+" : "+str(len(coredf.columns.values)))
        print(results[-1])

    dfresult=pd.DataFrame(results,columns=["COL","SCORE","STD","GAINMEAN","GAINSTD","RATIO","INCL"])
    return dfresult


#Given two sets of features, score each model and return a list of scores for each set
#Each Score in baseline should be comparable with the score of the same index in trnScore
#ie should be obtained with same crossvalidation data, model seed
def compareFeatureSets(df,target,featureSet1,featureSet2,scoreModel=models.scoreLGBModel,params=models.defaultLGBparams,seed=0):
    baseline=scoreModel(df[featureSet1],target,params,seed)['auc-fold']
    trnScore=scoreModel(df[featureSet2],target,params,seed)['auc-fold']
    return (baseline,trnScore)