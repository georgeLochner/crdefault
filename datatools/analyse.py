import datatools.datastore_util as du
import pandas as pd
import numpy as np
import pickle

# ranking=du.loadPickleDF("ranking")
# ranking2=du.loadPickleDF("ranking2")
# rankingFinal=du.loadPickleDF("rankingFinal")
# ranking3=du.loadPickleDF("ranking3")
# rankingFinal3=du.loadPickleDF("rankingFinal3")
# x=0
#
# path="./output/feature_ranks_2806/"
# ranking.to_csv(path+"ranking.csv")
# ranking2.to_csv(path+"ranking2.csv")
# rankingFinal.to_csv(path+"rankingFinal.csv")
# ranking3.to_csv(path+"ranking3.csv")
# rankingFinal3.to_csv(path+"rankingFinal3.csv")

# for index,row in ranking.iterrows():
#     line=""
#     for col in row:
#         line=line+str(row[col])+","
    # print(index+","+str(row["GAIN"])+","+str(row["STDV"]))


# goodFeats=ranking.iloc[0:10]
# features=[col for col in goodFeats.index]
# print(features)
#
# goodFeats=rankingFinal[rankingFinal["GAIN"]>0]
# features=[col for col in goodFeats.index]
# print(features)
#
# goodFeats=rankingFinal3[rankingFinal3["GAIN"]>0]
# features=[col for col in goodFeats.index]
# print(features)

def quickHack():
    """Tried to see if higher thresholds for feature inclusion in pruned data set, would result
    in last features added having improved contributions. Didnt seem so :?"""
    resultPath="./output/eval_prune_3006"
    totalauc = pd.Series()
    totalbl = pd.Series()
    v=[]
    last10=[]
    for i in range(0,5):
        file="result_"+"_"+str(i)
        result=du.loadPickleDF("result_" + str(i), path=resultPath)
        feats=result[result["INCL"]==True]["COL"].values
        print(file+":"+str(len(feats))+":"+str(result.iloc[-1]["SCORE"]))
        v.append(result.iloc[-1]["SCORE"])
        rl10=result.iloc[40:55]["RATIO"]
        last10.append(rl10.mean())
    print(str(np.mean(v))+":"+str(np.std(v)))
    print(str(np.mean(last10))+":"+str(np.std(last10)))
    print(str(totalauc.mean())+" "+str(totalauc.std())+" "+str(totalauc.mean()/totalauc.std()))

    resultPath="./output/eval_prune_3106"
    totalauc = pd.Series()
    totalbl = pd.Series()
    for th in (0.2,0.5,1.0):
        v=[]
        last10=[]
        for i in range(0,5):
            file="result_"+str(th)+"_"+str(i)
            result=du.loadPickleDF("result_" + str(th) +"_" + str(i), path=resultPath)
            feats=result[result["INCL"]==True]["COL"].values
            print(file+":"+str(len(feats))+":"+str(result.iloc[-1]["SCORE"]))
            v.append(result.iloc[-1]["SCORE"])
            rl10=result.iloc[40:55]["RATIO"]
            last10.append(rl10.mean())
        print(str(np.mean(v))+":"+str(np.std(v)))
        print(str(np.mean(last10))+":"+str(np.std(last10)))
    print(str(totalauc.mean())+" "+str(totalauc.std())+" "+str(totalauc.mean()/totalauc.std()))


#Relocated :) Compares the performance of the full dataset with a pruned feature dataset
def comparePrunedFeatureSet():
    resultPath="./output/"

    import pickle
    evalidx=int(len(train)*0.3)
    evaltrain=train[:evalidx]
    evaltarget=target[:evalidx]
    train=train[evalidx:]
    target=target[evalidx:]
    baseline=pd.Series()


    # for th in (0.2,0.5,1.0):
    #     for i in range(0,5):
    for th in (1.0,):
        for i in range(2,5):
            seed=randint(1, 60000)
            result=pd.DataFrame()
            result= fs.createRandomFeatureSet(train, target, params, scoreModel, seed, th)
            du.dfToPickle("result_"+str(th)+"_"+str(i),result,resultPath)
    gc.collect()

    pruned=[]
    baseline=[]
    for th in (0.2,0.5,1.0):
        for i in range(0,5):
            seed=randint(1, 60000)
            result=du.loadPickleDF("result_" + str(th) +"_" + str(i), path=resultPath)
            feats=result[result["INCL"]==True]["COL"].values
            baseline.append(evalLGBModelAUC(train,target,evaltrain,evaltarget,params,seed))
            pruned.append(evalLGBModelAUC(train[feats],target,evaltrain[feats],evaltarget,params,seed))
            pickle.dump({"baseline":baseline,"pruned":pruned}, open(resultPath+"eval_"+str(th)+"_"+str(i)+".pck", "wb"))
    print("------ done ")