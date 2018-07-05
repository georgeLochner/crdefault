import datatools.datastore_util as ds
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
import random

def compareBarPlot(col1,col2,col1name,col2name,title,ax=None):
    trncnt=col1.count()
    tstcnt=col2.count()
    trn:pd.Series=col1.value_counts()
    tst=col2.value_counts()
    tmp=trn.copy()
    if "Missing" in tmp.index: tmp["Missing"]=0
    adj=tmp.max()/(tst[tmp.idxmax()])
    tst=tst*adj
    #viz=pd.DataFrame([trn,tst],index=[col1name,col2name])
    viz=pd.DataFrame()
    viz[col1name]=trn
    viz[col2name]=tst
    viz=viz.transpose()
    ax=viz.plot(ax=ax,kind="barh",title=title,logx=True)
    #plt.show()
    #plt.close()
    return ax


def visualizeCategorical(df:pd.DataFrame,dataclasscol='dataclass'):
    for col in df:
        tp = df[col].dtype
        if ((str(df[col].dtype)=='category') & (col != dataclasscol)):
            print("Display "+col)
            trn=df[df[dataclasscol]=="Train"][col]
            tst=df[df[dataclasscol]=="Test"][col]
            if df[col].hasnans:
                trn=trn.cat.add_categories('Missing')
                tst=tst.cat.add_categories('Missing')
                trn=trn.fillna('Missing')
                tst=tst.fillna('Missing')
            fig=plt.figure(figsize=(10,10))
            ax1=fig.add_subplot(2,1,1)
            ax2=fig.add_subplot(2,1,2)
            compareBarPlot(trn,tst,"train","test",col,ax1)

            #plt.figure(figsize=(8,10))
            tgt=trn[(df[dataclasscol]=="Train") & (df["TARGET"]==1)]
            ctl=trn[(df[dataclasscol]=="Train") & (df["TARGET"]==0)]
            compareBarPlot(ctl,tgt,"control","target",None,ax2)
            plt.show()
            plt.close()

def visualizeNumerical(df:pd.DataFrame,dataclasscol='dataclass'):
    for col in df:
        tp = df[col].dtype
        if ((str(df[col].dtype)!='category') & (col != dataclasscol)):
            print("Display "+col)
            trn=df[df[dataclasscol]=="Train"][col]
            tst=df[df[dataclasscol]=="Test"][col]
            fig=plt.figure(figsize=(10,10))
            ax1=fig.add_subplot(2,1,1)
            ax2=fig.add_subplot(2,1,2)
            trn.plot(kind='hist',ax=ax1,logy=True,title=col)
            tst.plot(kind='hist',ax=ax2,logy=True)
            plt.show()
            plt.close()



def scatter3D(dfarr,Xcol,Ycol,Zcol,colors=['skyblue','red'],show=True):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for idx, df in enumerate(dfarr):
        ax.scatter(df[Xcol], df[Ycol], df[Zcol], c=colors[idx], s=20)
    ax.view_init(30, 185)

    # fig2=  plt.figure()
    # ax2 = fig2.add_subplot(111, projection='3d')
    # dffalse=df[df['TARGET']==0]
    # ax2.scatter(dffalse[Xcol], dffalse[Ycol], dffalse[Zcol], c='red', s=60)
    # ax2.view_init(30, 185)

    if show:plt.show()


def scatter2D(df, Xcol, Ycol, target="TARGET",show=True):
    sns.lmplot(Xcol,  # Horizontal axis
               Ycol,  # Vertical axis
               data=df,  # Data source
               fit_reg=False,  # Don't fix a regression line
               hue=target,  # Set color
               scatter_kws={"marker": "D", # Set marker style
                            "s": 100}) # S marker size

    if show: plt.show()



# visualizeCategorical(train)
#
#
# visualizeNumerical(train)

