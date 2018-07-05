import gc

import sys

import datatools.datastore_util as du
import datatools.model_util as modelUtil
import datatools.visualize as vis
import pandas as pd
import numpy as np
import datatools.preprocess_util as ut
import time
from datatools.transform_util import cart2sphere


def applicationPickle():
    # train = pd.read_csv("../data/csv/application_train.csv",encoding='latin1')
    # test = pd.read_csv("../data/csv/application_test.csv",encoding='latin1')

    train = pd.read_pickle('../data/pickle/application_train.pck')
    test = pd.read_pickle('../data/pickle/application_test.pck')

    train = train[train ['CODE_GENDER'] != 'XNA']
    train = train[train ['NAME_FAMILY_STATUS'] != 'Unknown']


    train['NAME_INCOME_TYPE'][train['NAME_INCOME_TYPE'] == 'Maternity leave']='Working'

    test["REGION_RATING_CLIENT_W_CITY"][test["REGION_RATING_CLIENT_W_CITY"]==-1]=1

    joined = du.joinTrainTestToH5(train,test)

    docs  = [_f for _f in train.columns if 'FLAG_DOC' in _f]
    joined['NEW_DOC_IND_KURT'] = joined[docs].kurtosis(axis=1)
    live = [_f for _f in train.columns if ('FLAG_' in _f) & ('FLAG_DOC' not in _f) & ('_FLAG_' not in _f)]
    joined['NEW_LIVE_IND_SUM'] = joined[live].sum(axis=1)

    #Classify major industries
    joined["ORGANIZATION_CAT"]=joined["ORGANIZATION_TYPE"].apply(lambda n : n.split(" ")[0] if not isinstance(n, float) else np.nan)
    ut.process_weekday(joined, "WEEKDAY_APPR_PROCESS_START", 'SATURDAY')


    dropFeats=["FLAG_DOCUMENT_2","FLAG_DOCUMENT_4","FLAG_DOCUMENT_7","FLAG_DOCUMENT_10","FLAG_DOCUMENT_12",
               "FLAG_DOCUMENT_13","FLAG_DOCUMENT_14","FLAG_DOCUMENT_15","FLAG_DOCUMENT_16","FLAG_DOCUMENT_17",
               "FLAG_DOCUMENT_19","FLAG_DOCUMENT_20","FLAG_DOCUMENT_21","FLAG_MOBIL"]

    joined.drop(dropFeats,axis=1,inplace=True)
    ut.process_categories(joined,["TARGET"])
    ut.raiseIfUselessFeatures(du.getTrainDF(joined))



    joined["DAYS_EMPLOYED"].replace(365243, np.nan, inplace= True)
    ut.addNanColumns(joined,["DAYS_EMPLOYED"])
    ut.outliersToMax(joined,"DEF_60_CNT_SOCIAL_CIRCLE",12)
    ut.outliersToMax(joined,"DEF_30_CNT_SOCIAL_CIRCLE",12)
    ut.outliersToMax(joined,"OBS_30_CNT_SOCIAL_CIRCLE",50)
    ut.outliersToMax(joined,"OBS_60_CNT_SOCIAL_CIRCLE",50)
    ut.outliersToMax(joined,"AMT_REQ_CREDIT_BUREAU_QRT",10)
    ut.outliersToMax(joined,"OBS_60_CNT_SOCIAL_CIRCLE",50)
    ut.outliersToMax(joined,"AMT_INCOME_TOTAL",100000000)

    rez=cart2sphere(joined[['AMT_ANNUITY','AMT_CREDIT','AMT_GOODS_PRICE']])
    joined['ACP_R']=rez['r']
    joined['ACP_EL']=rez['el']
    joined['ACP_Z']=rez['az']

# inc_by_org = joined[['AMT_INCOME_TOTAL', 'ORGANIZATION_TYPE']].groupby('ORGANIZATION_TYPE').median()['AMT_INCOME_TOTAL']
    # joined['NEW_CREDIT_TO_ANNUITY_RATIO'] = joined['AMT_CREDIT'] / joined['AMT_ANNUITY']
    # joined['NEW_CREDIT_TO_GOODS_RATIO'] = joined['AMT_CREDIT'] / joined['AMT_GOODS_PRICE']
    # joined['NEW_INC_PER_CHLD'] = joined['AMT_INCOME_TOTAL'] / (1 + joined['CNT_CHILDREN'])
    # joined['NEW_INC_BY_ORG'] = joined['ORGANIZATION_TYPE'].map(inc_by_org)
    # joined['NEW_EMPLOY_TO_BIRTH_RATIO'] = joined['DAYS_EMPLOYED'] / joined['DAYS_BIRTH']
    # joined['NEW_ANNUITY_TO_INCOME_RATIO'] = joined['AMT_ANNUITY'] / (1 + joined['AMT_INCOME_TOTAL'])
    # joined['NEW_SOURCES_PROD'] = joined['EXT_SOURCE_1'] * joined['EXT_SOURCE_2'] * joined['EXT_SOURCE_3']
    # joined['NEW_EXT_SOURCES_MEAN'] = joined[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].mean(axis=1)
    # joined['NEW_SCORES_STD'] = joined[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']].std(axis=1)
    # joined['NEW_SCORES_STD'] = joined['NEW_SCORES_STD'].fillna(joined['NEW_SCORES_STD'].mean())
    # joined['NEW_CAR_TO_BIRTH_RATIO'] = joined['OWN_CAR_AGE'] / joined['DAYS_BIRTH']
    # joined['NEW_CAR_TO_EMPLOY_RATIO'] = joined['OWN_CAR_AGE'] / joined['DAYS_EMPLOYED']
    # joined['NEW_PHONE_TO_BIRTH_RATIO'] = joined['DAYS_LAST_PHONE_CHANGE'] / joined['DAYS_BIRTH']
    # joined['NEW_PHONE_TO_EMPLOY_RATIO'] = joined['DAYS_LAST_PHONE_CHANGE'] / joined['DAYS_EMPLOYED']
    # joined['NEW_CREDIT_TO_INCOME_RATIO'] = joined['AMT_CREDIT'] / joined['AMT_INCOME_TOTAL']
    #
    #
    # joined['DAYS_EMPLOYED_PERC'] = joined['DAYS_EMPLOYED'] / joined['DAYS_BIRTH']
    # joined['INCOME_CREDIT_PERC'] = joined['AMT_INCOME_TOTAL'] / joined['AMT_CREDIT']
    # joined['INCOME_PER_PERSON'] = joined['AMT_INCOME_TOTAL'] / joined['CNT_FAM_MEMBERS']
    # joined['ANNUITY_INCOME_PERC'] = joined['AMT_ANNUITY'] / joined['AMT_INCOME_TOTAL']
    # joined['PAYMENT_RATE'] = joined['AMT_ANNUITY'] / joined['AMT_CREDIT']

    du.dfToPickle("application",joined)

# One-hot encoding for categorical columns with get_dummies
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns


def bureauBalancePickle(num_rows = None, nan_as_category = True):

    bureau = pd.read_pickle('../data/pickle/bureau.pck')
    bb = pd.read_pickle('../data/pickle/bureau_balance.pck')


    bb, bb_cat = one_hot_encoder(bb, nan_as_category)
    bureau, bureau_cat = one_hot_encoder(bureau, nan_as_category)

    bureau=bureau[bureau["DAYS_CREDIT_UPDATE"]>-180]

    # Bureau balance: Perform aggregations and merge with bureau.csv
    bb_aggregations = {'MONTHS_BALANCE': ['min', 'max', 'size']}
    for col in bb_cat:
        bb_aggregations[col] = ['mean']
    bb_agg = bb.groupby('SK_ID_BUREAU').agg(bb_aggregations)
    bb_agg.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bb_agg.columns.tolist()])
    bureau = bureau.join(bb_agg, how='left', on='SK_ID_BUREAU')
    bureau.drop(['SK_ID_BUREAU'], axis=1, inplace= True)
    del bb, bb_agg
    gc.collect()

    # Bureau and bureau_balance numeric features
    num_aggregations = {
        'DAYS_CREDIT': ['min', 'max', 'mean', 'var'],
        'DAYS_CREDIT_ENDDATE': ['min', 'max', 'mean'],
        'DAYS_CREDIT_UPDATE': ['mean'],
        'CREDIT_DAY_OVERDUE': ['max', 'mean'],
        'AMT_CREDIT_MAX_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_DEBT': ['max', 'mean', 'sum'],
        'AMT_CREDIT_SUM_OVERDUE': ['mean'],
        'AMT_CREDIT_SUM_LIMIT': ['mean', 'sum'],
        'AMT_ANNUITY': ['max', 'mean'],
        'CNT_CREDIT_PROLONG': ['sum'],
        'MONTHS_BALANCE_MIN': ['min'],
        'MONTHS_BALANCE_MAX': ['max'],
        'MONTHS_BALANCE_SIZE': ['mean', 'sum']
    }
    # Bureau and bureau_balance categorical features
    cat_aggregations = {}
    for cat in bureau_cat: cat_aggregations[cat] = ['mean']
    for cat in bb_cat: cat_aggregations[cat + "_MEAN"] = ['mean']

    bureau_agg = bureau.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})
    bureau_agg.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper() for e in bureau_agg.columns.tolist()])
    # Bureau: Active credits - using only numerical aggregations
    active = bureau[bureau['CREDIT_ACTIVE_Active'] == 1]
    active_agg = active.groupby('SK_ID_CURR').agg(num_aggregations)
    active_agg.columns = pd.Index(['ACTIVE_' + e[0] + "_" + e[1].upper() for e in active_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(active_agg, how='left')
    del active, active_agg
    gc.collect()
    # Bureau: Closed credits - using only numerical aggregations
    closed = bureau[bureau['CREDIT_ACTIVE_Closed'] == 1]
    closed_agg = closed.groupby('SK_ID_CURR').agg(num_aggregations)
    closed_agg.columns = pd.Index(['CLOSED_' + e[0] + "_" + e[1].upper() for e in closed_agg.columns.tolist()])
    bureau_agg = bureau_agg.join(closed_agg, how='left')
    del closed, closed_agg, bureau
    gc.collect()
    joined=du.loadPickleDF("application")
    joined=joined[["SK_ID_CURR","TARGET","dataclass"]]
    bureau_agg = joined.join(bureau_agg, how='left',on="SK_ID_CURR")
    du.dfToPickle("bureau",bureau_agg)



def bureauBalanceLoanPredictPickle(num_rows = None, nan_as_category = True):
    # bureau = pd.read_csv('../data/csv/bureau.csv', nrows = num_rows)
    # bureau.to_pickle('../data/csv/bureau.pkl')
    bureau = pd.read_pickle('../data/pickle/bureau.pck')
    # bb = pd.read_csv('../data/csv/bureau_balance.csv', nrows = num_rows)
    # bb.to_pickle('../data/csv/bureau_balance.pkl')
    bb = pd.read_pickle('../data/pickle/bureau_balance.pck')
    joined=du.loadPickleDF("application")
    joined=joined[["SK_ID_CURR","TARGET","dataclass"]]
    # bureau = joined.merge(bureau, how='left',on="SK_ID_CURR")
    bureau = bureau.merge(joined, how='left',on="SK_ID_CURR")
    bureau = bureau[bureau["dataclass"]=="Train"]
    bureau=bureau.drop(["dataclass"],axis=1)
    ut.process_categories(bureau,["TARGET"],checkCats=False)
    du.dfToPickle("bureausmall-loanpredict",bureau)
    print("done")

def prepareSmallPickle():
    joined=du.loadPickleDF("application")
    small=joined[joined["dataclass"]=="Train"].iloc[:]
    small=small.drop(["dataclass"],axis=1)
    du.dfToPickle("small",small)
    skid=small[["SK_ID_CURR"]]
    du.dfToPickle("skidsmall",skid)
    bbsmall = du.loadPickleDF("bureau")
    bbsmall = skid.merge(bbsmall, how='left',on='SK_ID_CURR')
    bbsmall =bbsmall.drop(["dataclass"],axis=1)
    du.dfToPickle("bureausmall",bbsmall)

    print("done")

# du.csvdirToPickle("../data/csv")
# applicationPickle()
# bureauBalancePickle()
# prepareSmallPickle()
bureauBalanceLoanPredictPickle()
debug=False
numBoostRounds=5000
coreMax = 15

# train = du.loadPickle("traintest")
# train = du.loadPickleDF("small")
# train = du.loadPickleDF("bureausmall")
train = du.loadPickleDF("bureausmall-loanpredict")

# feats=["TARGET","dataclass",'BURO_DAYS_CREDIT_ENDDATE_MIN' ,'BURO_DAYS_CREDIT_ENDDATE_MAX', 'BURO_DAYS_CREDIT_ENDDATE_MEAN','ACTIVE_DAYS_CREDIT_ENDDATE_MIN', 'ACTIVE_DAYS_CREDIT_ENDDATE_MAX', 'ACTIVE_DAYS_CREDIT_ENDDATE_MEAN', 'CLOSED_DAYS_CREDIT_ENDDATE_MIN' ,'CLOSED_DAYS_CREDIT_ENDDATE_MAX', 'CLOSED_DAYS_CREDIT_ENDDATE_MEAN','BURO_MONTHS_BALANCE_MIN_MIN', 'BURO_MONTHS_BALANCE_MAX_MAX' ,'BURO_MONTHS_BALANCE_SIZE_MEAN', 'BURO_MONTHS_BALANCE_SIZE_SUM', 'ACTIVE_MONTHS_BALANCE_MIN_MIN', 'ACTIVE_MONTHS_BALANCE_MAX_MAX', 'ACTIVE_MONTHS_BALANCE_SIZE_MEAN','ACTIVE_MONTHS_BALANCE_SIZE_SUM','CLOSED_MONTHS_BALANCE_MIN_MIN', 'CLOSED_MONTHS_BALANCE_MAX_MAX' ,'CLOSED_MONTHS_BALANCE_SIZE_MEAN', 'CLOSED_MONTHS_BALANCE_SIZE_SUM',]
# feats=["TARGET","dataclass",'BURO_CNT_CREDIT_PROLONG_SUM','ACTIVE_CNT_CREDIT_PROLONG_SUM','CLOSED_CNT_CREDIT_PROLONG_SUM']
# train = train[feats]
# vis.visualizeCategorical(train)
# vis.visualizeNumerical(train)
# sys.exit(0)

if (debug):
    #train = train.iloc[:6000]
    train = train[["SK_ID_CURR","TARGET","DAYS_BIRTH","AMT_GOODS_PRICE","AMT_ANNUITY","DAYS_EMPLOYED","CODE_GENDER","DAYS_ID_PUBLISH"]]
    numBoostRounds=500
    coreMax=3

# def testfunc(df,a='',b=''):
#     return 0
# aggtest = train.groupby('ORGANIZATION_TYPE').agg({'AMT_INCOME_TOTAL':testfunc})


target=train["TARGET"]
train.drop("SK_ID_CURR",axis=1,inplace=True)
train.drop("SK_ID_BUREAU",axis=1,inplace=True)
train.drop("TARGET",axis=1,inplace=True)


from sklearn.metrics import mean_absolute_error


params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric':'auc',
    'learning_rate': 0.1, #learning rate: keep it only for training speed, not to tune (otherwise overfitting)
    'num_leaves': 7,  # [7, 4095] we should let it be smaller than 2^(max_depth)
    'max_depth': 3,  #[2, 63] -1 means no limit
    'colsample_bytree': 1,  #1.0 [0.4, 1] Subsample ratio of columns when constructing each tree.
    'reg_alpha':0.041545473,
              'reg_lambda':0.0735294,
    # 'subsample': 0.4,  #[0.4, 1] bagging_fraction Subsample ratio of the training instance.
    # 'subsample_freq': 1,  #bagging_freq frequence of subsample, <=0 means no enable
}

print("Training...")
start_time = time.time()


# seed=randint(0, 30000)
# # 'EXT_SOURCE_2','EXT_SOURCE_3','EXT_SOURCE_1','REGION_RATING_CLIENT_W_CITY',
# train1=train[['REGION_RATING_CLIENT']]
# result1= scoreModel(train1,target,params,seed)
# train2=train[['REGION_RATING_CLIENT','REGION_RATING_CLIENT_W_CITY']]
# result2= scoreModel(train2,target,params,seed)
# aucdiff=result2['auc-fold']-result1['auc-fold']
# print(result1['auc-fold'].mean())
# print(result1['auc-fold'].std())
# print(result2['auc-fold'].mean())
# print(result2['auc-fold'].std())
# print(aucdiff.mean())
# print(aucdiff.std())
# print(aucdiff.mean()/aucdiff.std())


# folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=546789)
# train=train[['EXT_SOURCE_2', 'EXT_SOURCE_3', 'CODE_GENDER', 'EXT_SOURCE_1', 'AMT_GOODS_PRICE', 'DAYS_BIRTH', 'AMT_CREDIT', 'AMT_ANNUITY', 'DAYS_EMPLOYED',
#  'DAYS_ID_PUBLISH','NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NONLIVINGAPARTMENTS_MODE', 'NEW_DOC_IND_KURT', 'OWN_CAR_AGE', 'FLOORSMAX_MODE',
#  'AMT_REQ_CREDIT_BUREAU_MON', 'OCCUPATION_TYPE', 'DEF_30_CNT_SOCIAL_CIRCLE', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'REGION_RATING_CLIENT_W_CITY', 'AMT_INCOME_TOTAL',
#  'DAYS_EMPLOYED_NAN', 'FLAG_EMP_PHONE', 'NONLIVINGAPARTMENTS_AVG', 'FLOORSMIN_AVG', 'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_HOUR',
#  'NEW_CREDIT_TO_ANNUITY_RATIO', 'PAYMENT_RATE', 'INCOME_CREDIT_PERC', 'NEW_CREDIT_TO_GOODS_RATIO', 'NEW_CREDIT_TO_INCOME_RATIO']]
# result = modelUtil.train_lgb_cv(train,  target, params,saveData=True)
# print("Done")


#opt.findUsefulFeatures(train,target,params,scoreModel,0.001,15)
# newFeats=["DAYS_ID_PUBLISH","AMT_ANNUITY"]

# newFeats=['PAYMENT_RATE','ANNUITY_INCOME_PERC','INCOME_PER_PERSON','INCOME_CREDIT_PERC','DAYS_EMPLOYED_PERC','NEW_CREDIT_TO_INCOME_RATIO','NEW_PHONE_TO_EMPLOY_RATIO','NEW_PHONE_TO_BIRTH_RATIO','NEW_CAR_TO_EMPLOY_RATIO','NEW_CAR_TO_BIRTH_RATIO','NEW_SCORES_STD','NEW_EXT_SOURCES_MEAN','NEW_SOURCES_PROD','NEW_ANNUITY_TO_INCOME_RATIO','NEW_EMPLOY_TO_BIRTH_RATIO','NEW_INC_BY_ORG','NEW_INC_PER_CHLD','NEW_CREDIT_TO_ANNUITY_RATIO','NEW_CREDIT_TO_GOODS_RATIO']
# opt.findUsefulFeatures(train,target,params,scoreModel,newFeats,.001,coreMax)
#


# #
# #
# #
# result,bst = modelUtil.scoreLGBModel(train[['RATIO']], target, params, seed=0,early_stop=200)
result,bst = modelUtil.scoreLGBModel(train, target, params, seed=0,early_stop=100)
print(result['auc-fold'].mean())
print(result['auc-fold'].std())
print('Model training time : {} '.format(time.time() - start_time))




