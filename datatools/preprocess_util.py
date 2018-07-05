import pandas as pd
from sklearn.preprocessing import LabelEncoder

def raiseIfUselessFeatures(df :pd.DataFrame, excludeCat=[], threshold=100):
    for colname in list(df):
        if ((colname not in excludeCat) & (df[colname].dtype in ["object","int64"] )):
            vc=df[colname].value_counts()
            minorTotal=len(df[colname])-vc.max()
            if (minorTotal<threshold):
                print(df[colname].value_counts())
                raise Exception(colname +" contains fewer than "+str(threshold)+" unique values")

def addNanColumns(df:pd.DataFrame,colnames):
    for colname in colnames:
        if df[colname].hasnans:
            df[colname+"_NAN"]=pd.isnull(df[colname]).astype(int)
            df[colname+"_NAN"]=df[colname+"_NAN"].astype('category')
            print("Adding column "+colname+"_NAN")

#Convert columns with discrete values to categorical
#NB Object columns will convert to integer encoded categorical and convert Nan to new category
#NB Integer columns assume ordering and will convert Nan to new column (ie col indicates has value or not) if nanToCol=True
def process_categories(df :pd.DataFrame,excludeCat=[],factorize=True,threshold=100,nanToCol=True,checkCats=True):
    if checkCats:
        trnidx=df[df["dataclass"]=="Train"].index
        tstidx=df[df["dataclass"]=="Test"].index
    def printFail(col1,col2,msg):
        print(col1.value_counts())
        print(col2.value_counts())
        raise Exception(msg)
    def checkCatTrainTest(col):
        if (col[tstidx].value_counts()==0).any():
            printFail(col[trnidx],col[tstidx],"Test set is missing category")
        if (col[trnidx].value_counts()==0).any():
            printFail(col[trnidx],col[tstidx],"Train set is missing category")
    converted=[]
    cols=df.dtypes[df.dtypes=='object'].index
    for colname in cols:
        if ((df[colname].nunique() < threshold) & (colname not in excludeCat)):
            print("Converting to category:"+colname)
            df[colname]=df[colname].astype('category')
            if checkCats:checkCatTrainTest(df[colname])
            if factorize:
                df[colname]=pd.factorize(df[colname],na_sentinel=df[colname].nunique())[0]
                df[colname]=df[colname].astype('category')
            converted.append(colname)
    cols=df.dtypes[df.dtypes=='int64'].index
    for colname in cols:
        if ((df[colname].nunique() <10) and (colname not in excludeCat)):
            print("Converting to category:"+colname)
            df[colname]=df[colname].astype('category')
            if df[colname].nunique() >2:
                df[colname].cat.as_ordered(inplace=True)
            if checkCats:checkCatTrainTest(df[colname])
            converted.append(colname)
            if nanToCol: addNanColumns(df,[colname])

    return converted


#Convert days of the week to integers, setting an arbitrary day as 1
def process_weekday(df,colname,weekStartDay='MONDAY', addWkendFlagCol=True):
    vals=df[colname].value_counts()
    weekdays=['MONDAY','TUESDAY','WEDNESDAY','THURSDAY','FRIDAY','SATURDAY','SUNDAY']
    for v in vals.index:
        if v.upper() not in weekdays: raise "Column "+colname+" contains non weekday entries"
    df[colname]=df[colname].apply(lambda d: weekdays.index(d.upper())+1)
    startweek=weekdays.index(weekStartDay)+1
    if addWkendFlagCol:
        df[colname+'_WKENDFLAG']=((df[colname] == 6) | (df[colname] == 7))
        df[colname]=df[colname].apply(lambda d: d-startweek+8 if d<startweek else d-startweek+1)
    return None

def outliersToMax(df:pd.DataFrame,colname,threshold):
    max = df[colname][df[colname] < threshold].max()
    df[colname][df[colname] > threshold] = max