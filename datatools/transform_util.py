import math
from inspect import signature, getmembers
from random import randint

from sklearn import preprocessing

import datatools.feature_selection as fs
import datatools.datastore_util as du
import datatools.visualize as viz
import pandas as pd
import numpy as np
import lightgbm as lgb
import time
import matplotlib.pyplot as plt
import seaborn as sns

import datatools.model_util as mu

# train = du.loadPickleDF("small")
# target=train["TARGET"]
# train.drop("SK_ID_CURR",axis=1,inplace=True)
# train.drop("TARGET",axis=1,inplace=True)

params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric':'auc',
    'learning_rate': 0.1, #learning rate: keep it only for training speed, not to tune (otherwise overfitting)
    'num_leaves': 7,  # [7, 4095] we should let it be smaller than 2^(max_depth)
    'max_depth': 3,  #[2, 63] -1 means no limit
    'colsample_bytree': 0.9,  #1.0 [0.4, 1] Subsample ratio of columns when constructing each tree.
}
# log(a) , ratio(a,b) ,



def ratio(df):
    result = df[df.columns[0]]/df[df.columns[1]]
    return pd.DataFrame({'col1/col2':result})

def subtract(df):
    result = df[df.columns[0]]-df[df.columns[1]]
    return pd.DataFrame({'col1-col2':result})

def log(a):
    a[a==0]=np.nan
    a=a.apply(math.fabs)
    return a.apply(math.log10)


def cart2pol(cartdf):
    cartdf = cartdf/cartdf.abs().max()
    rho = np.sqrt(cartdf[cartdf.columns[0]]**2 + cartdf[cartdf.columns[1]]**2)
    phi = np.arctan2(cartdf[cartdf.columns[1]], cartdf[cartdf.columns[0]])
    return pd.DataFrame({"rho":rho, "phi":phi})

def cart2sphere(xyzdf) -> pd.DataFrame:
    x=xyzdf[xyzdf.columns[0]]
    y=xyzdf[xyzdf.columns[1]]
    z=xyzdf[xyzdf.columns[2]]
    XsqPlusYsq = x**2 + y**2
    r = np.sqrt(XsqPlusYsq + z**2)               # r
    elev = np.arctan2(z,np.sqrt(XsqPlusYsq))     # theta
    az = np.arctan2(y,x)                           # phi
    return pd.DataFrame({"r":r, "el":elev, "az":az})

def getFuncName(func):
    for attr, funcname in getmembers(func):
        if attr == '__name__': break
    return funcname

#Deprecated function to evaluaute transforms on a single feature
def evalFeatureSingle(train:pd.DataFrame,target:pd.DataFrame,transformations,scoreModel,params,features):
    result={}
    for func in transformations:
        funcname=getFuncName(func)
        paramcnt = len(signature(func).parameters)
        result[funcname]={}
        summary = pd.DataFrame(columns=["Mean","Std","Ratio"])
        for f1 in features:
            print("------"+f1+"---------")
            result[funcname][f1]={}
            start = time.time()
            seed=randint(1, 60000)
            tmpdf=pd.DataFrame(train[f1])
            newcol=funcname+f1
            tmpdf[newcol]=func(train[f1])
            baseline=scoreModel(tmpdf[[f1]],target,params,seed)['auc-fold']
            trnScore=scoreModel(tmpdf,target,params,seed)['auc-fold']
            result[funcname][f1]['bl']=baseline
            result[funcname][f1]['trn']=trnScore
            end = time.time()
            result[funcname][f1]['tm']=end-start
            du.dictToPickle("./output/trn_"+funcname+".dict",result)
            diff=trnScore-baseline
            summary.loc[f1]=((diff.mean(),diff.std(),diff.mean()/diff.std()))
        du.dfToPickle("trn_" + funcname +"_summary", summary, outputPath="./output")
        print(params)

#Compare the performance of a list of transformations for each pair of features in the 'features' list
def transformPair(train:pd.DataFrame,target:pd.DataFrame,transformations,features,scoreModel=mu.scoreLGBModel,params=mu.defaultLGBparams,bidirectional=False
                  ,resultFile="transform_summary",resultPath="./output"):
    result={}
    for func in transformations:
        funcname=getFuncName(func)
        result[funcname]={}
        summary = pd.DataFrame(columns=["Feat1","Feat2","Baseline","Transf","Mean","Std","Ratio","Func"])
        for f1 in features:
            for f2 in features:
                if f1 == f2:continue
                if not bidirectional and ((summary[(summary['Feat1']==f2) & (summary['Feat2']==f1)])>0): continue
                print("------"+f1+"-"+f2+"---------")
                seed=randint(1, 60000)
                fs1=[f1,f2]
                cart=train[fs1]
                pol=func(cart)
                fs2=fs1.copy()
                for col in pol:
                    newfeat=funcname+"-"+col
                    fs2.append(newfeat)
                    cart[newfeat]=pol[col]
                baseline,trnScore = fs.compareFeatureSets(cart,target,fs1,fs2,scoreModel,params,seed)
                diff=trnScore-baseline
                summary.loc[len(summary)]=(f1,f2,baseline.mean(),trnScore.mean(),diff.mean(),diff.std(),diff.mean()/diff.std(),funcname)
                du.dfToPickle(resultFile, summary, outputPath=resultPath)
            print(summary)
    return summary

# features=train.select_dtypes(include=['float64','int64']).columns.values
# # features=[f for f in features if not f.endswith("_MEDI")]
# # features=[f for f in features if not f.endswith("_MODE")]
# features=['ELEVATORS_AVG','ELEVATORS_MEDI','ELEVATORS_MODE']

# Visualize PCA transform
# from sklearn import decomposition
# pcadf=train[features]
# pcadf=pcadf.dropna()
# pca = decomposition.FastICA(n_components=3)
# pca.fit(pcadf)
# X = pca.transform(pcadf[0:3000])
# Xdf=pd.DataFrame(X,columns=['ELEVATORS_AVG','ELEVATORS_MEDI','ELEVATORS_MODE'])
# viz.scatter3D([Xdf],'ELEVATORS_AVG','ELEVATORS_MEDI','ELEVATORS_MODE',show=False)
# viz.scatter3D([pcadf[0:3000]],'ELEVATORS_AVG','ELEVATORS_MEDI','ELEVATORS_MODE')

# c=train[features].corr()
# transformPair(train,target,[cart2pol],params,features,scoreModel)
# transformPair(train,target,[subtract],params,features,scoreModel,bidirectional=True)

# pol=cart2pol(train[['ENTRANCES_AVG','ENTRANCES_MEDI']])
# train["phi"]=pol["phi"]
# train["rho"]=pol["rho"]
# sns.lmplot('phi', # Horizontal axis
#            'rho', # Vertical axis
#            data=train, # Data source
#            fit_reg=False, # Don't fix a regression line
#            hue="TARGET", # Set color
#            scatter_kws={"marker": "D", # Set marker style
#                         "s": 100}) # S marker size

# sns.lmplot('REGION_RATING_CLIENT', # Horizontal axis
#            'AMT_INCOME_TOTAL', # Vertical axis
#            data=train, # Data source
#            fit_reg=False, # Don't fix a regression line
#            hue="TARGET", # Set color
#            scatter_kws={"marker": "D", # Set marker style
#                         "s": 100}) # S marker size
#
# plt.show()

# trainfalse=train[train["TARGET"]==0][0:3000]
# traintrue=train[train["TARGET"]==1][0:3000]
# # viz.scatter3D([trainfalse,traintrue],'AMT_ANNUITY','AMT_CREDIT','AMT_GOODS_PRICE')
# viz.scatter3D([trainfalse,traintrue],'ENTRANCES_AVG','ENTRANCES_MEDI','ENTRANCES_MODE')
# # viz.scatter3D([trainfalse,traintrue],'EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3')

## Spherical transform test 'AMT_ANNUITY','AMT_CREDIT','AMT_GOODS_PRICE'
# fs1=['AMT_ANNUITY','AMT_CREDIT','AMT_GOODS_PRICE','EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH','CODE_GENDER']
# df=train[fs1]
# sph=cart2sphere(df)
# fs2=list(sph.columns.values)
# fs2.extend(['EXT_SOURCE_1','EXT_SOURCE_2','EXT_SOURCE_3','DAYS_BIRTH','CODE_GENDER'])
# df=pd.concat([df,sph],axis=1)
# summary = pd.DataFrame(columns=["Baseline","Transf","Mean","Std","Ratio"])
# baseline,trnScore = compareFeatureSets(df,target,fs1,fs2,scoreModel,params,0)
# diff=trnScore-baseline
# summary.loc[len(summary)]=(baseline.mean(),trnScore.mean(),diff.mean(),diff.std(),diff.mean()/diff.std())
# print(summary)

#PCA Transform test  'ENTRANCES_AVG','ENTRANCES_MEDI','ENTRANCES_MODE'
# train['TARGET']=target
# features=['TARGET','ELEVATORS_AVG','ELEVATORS_MEDI','ELEVATORS_MODE','APARTMENTS_AVG','APARTMENTS_MEDI','APARTMENTS_MODE']
# scale = train['ELEVATORS_AVG'].mean()/train['APARTMENTS_AVG'].mean()
# train['APARTMENTS_AVG']=train['APARTMENTS_AVG']*scale;
# train['APARTMENTS_MEDI']=train['APARTMENTS_MEDI']*scale;
# train['APARTMENTS_MODE']=train['APARTMENTS_MODE']*scale;
# summary = pd.DataFrame(columns=["Baseline","Transf","Mean","Std","Ratio"])
# from sklearn import decomposition
# for i in range(10):
#     pcadf=train[features]
#     pcadf=pcadf.dropna()
#     target=pcadf['TARGET']
#     pcadf=pcadf.drop('TARGET',axis=1)
#     pca = decomposition.FastICA(n_components=6)
#     pca.fit(pcadf)
#     X = pca.transform(pcadf)
#     Xdf=pd.DataFrame(X,columns=['TX1','TX2','TX3','TX4','TX5','TX6'],index=pcadf.index)
#     df=pd.concat([pcadf,Xdf],axis=1)
#     baseline,trnScore = compareFeatureSets(df,target,['ELEVATORS_AVG','ELEVATORS_MEDI','ELEVATORS_MODE','APARTMENTS_AVG','APARTMENTS_MEDI','APARTMENTS_MODE'],['TX1','TX2','TX3','TX4','TX5','TX6'],scoreModel,params,randint(1, 60000))
#     diff=trnScore-baseline
#     summary.loc[len(summary)]=(baseline.mean(),trnScore.mean(),diff.mean(),diff.std(),diff.mean()/diff.std())
# print(summary)
# print(summary.mean())
# print("Done")