import glob
import pandas as pd
import numpy as np
import os
import pickle
import ntpath
defaultDataPath= "../data/pickle"
defaultOutputPath= "../data"

#Convert csvs to H5 format
#Object data types with fewer than cat_thresh unique values are converted to categories
#excludeCat - columns to exclude from categorization
def dfToPickle(filename, df:pd.DataFrame, outputPath=defaultOutputPath):
    # ds=filename+".h5"
    # ds=os.path.join(h5path, ds)
    # df.to_hdf(ds,mode='w',key='data',format='t')
    pds=filename+".pck"
    pds=os.path.join(outputPath, pds)
    df.to_pickle(pds)
    print("created "+pds)


def csvdirToPickle(csvpath, outputPath=defaultDataPath):
    if outputPath==None:outputPath=csvpath
    srch=os.path.join(csvpath, "*.csv")
    csvs=glob.glob(srch, recursive=False)
    for f in csvs:
        df = pd.read_csv(f)
        fname = ntpath.basename(f)
        fname = os.path.splitext(fname)[0]+".pck"
        pck=os.path.join(outputPath, fname)
        df.to_pickle(pck)

def dictToPickle(filename,dict):
    pickle.dump(dict, open(filename, "wb"))

def joinTrainTestToH5(traindf,testdf,targetcol="TARGET") ->pd.DataFrame:
    testdf[targetcol] = np.nan
    # def printFail(col1,col2,msg):
    #     print(col1.value_counts())
    #     print(col2.value_counts())
    #     raise Exception(msg)
    # for col in traindf:
    #     if ((str(traindf[col].dtype)=='category')):
    #         print("Checking "+col)
    #         for c in  traindf[col].cat.categories:
    #             if c not in testdf[col].cat.categories:
    #                 printFail(traindf[col],testdf[col],"Test set is missing category "+str(c)+" for "+col)
    #         for c in  testdf[col].cat.categories:
    #             if c not in traindf[col].cat.categories:
    #                 printFail(traindf[col],testdf[col],"Train set is missing category "+str(c)+" for "+col)
    traindf["dataclass"]="Train"
    testdf["dataclass"]="Test"
    wtf:pd.DataFrame=pd.concat([traindf,testdf],ignore_index=True)

    wtf["dataclass"]=wtf["dataclass"].astype("category")
    return wtf

def getTrainDF(joinedDF) -> pd.DataFrame:
    return joinedDF[joinedDF["dataclass"]=="Train"]
def getTestDF(joinedDF) -> pd.DataFrame:
    return joinedDF[joinedDF["dataclass"]=="Test"]


def splitData(df:pd.DataFrame, h5Name, rows=50000, h5path=defaultDataPath):
    splits=int(round(df.shape[0]/rows,0))
    rowsPerSplit = int(round(df.shape[0]/splits,0))
    r=0
    cnt=1
    while (r<df.shape[0]):
        last=r+rowsPerSplit
        if last > df.shape[0]:last=df.shape[0]
        tmp=df.iloc[r:last]
        ds=os.path.join(h5path, h5Name+str(cnt)+".h5")
        tmp.to_hdf(ds,mode='w',key='data',format='t')
        r=last
        cnt+=1



def extract(df:pd.DataFrame, keyCol, keyList, h5Name, h5path=defaultDataPath):
    return None

def loadH5(datafile, h5path=defaultDataPath) -> pd.DataFrame:
    ds=os.path.join(h5path, datafile+".h5")
    store=pd.HDFStore(ds)
    data= store['data']
    store.close()
    return data

def loadPickleDF(datafile, path=defaultOutputPath) -> pd.DataFrame:
    ds=os.path.join(path, datafile + ".pck")
    data = pickle.load( open( ds, "rb" ) )
    return data

def loadPickle(datafile, path=defaultOutputPath):
    ds=os.path.join(path, datafile + ".pck")
    data = pickle.load( open( ds, "rb" ) )
    return data

# data= pd.HDFStore('../data/h5/application_train.h5')['data']
# splitData(data,"train")

# csvdirToH5("../data/csv","../data/h5")
#
#joinTrainTestToH5("application_train","application_test","appl")

#csvToH5("../data/csv/application_test.csv","../data/h5")
