import pandas as pd

data_dir = "../data/"


def getShalla():
    """load shalla dataset"""
    print("Shalla_Cost reading...")
    data=pd.read_csv(data_dir+"shalla_cost.txt", sep=' ', names=['type','ID', 'cost'], header=None)
    positives = data.loc[data['type']==1].iloc[:,1].values
    negatives = data.loc[data['type']==0].iloc[:,1].values
    # neg_cost = data.loc[data['type']==0].iloc[:,2].values
    print("positives size: ", len(positives))
    print("negatives size: ", len(negatives))
    return positives, negatives, neg_cost


def get_urldata():
    """load urldata dataset"""
    print("urldata reading...")
    data = pd.read_csv(data_dir + "pre_process_url.txt", sep='\t', names=['type', 'ID'], header=None)
    positives = data.loc[data['type'] == 1].iloc[:, 1].values
    negatives = data.loc[data['type'] == 0].iloc[:, 1].values
    print("positives size: ", len(positives))
    print("negatives size: ", len(negatives))
    return positives, negatives


def getYCSB():
    """load ycsb dataset"""
    print("YCSB reading...")
    data=pd.read_csv(data_dir+"ycsbt.txt", sep=' ', names=['type','ID', 'cost'], header=None)
    positives = data.loc[data['type']=="FILTERKEY"].iloc[:,1].values
    negatives = data.loc[data['type']=="OTHERKEY"].iloc[:,1].values
    neg_cost = data.loc[data['type']=="OTHERKEY"].iloc[:,2].values
    print("positives size: ", len(positives))
    print("negatives size: ", len(negatives))
    return positives, negatives, neg_cost


def get_ultimate():
    """load ultimate dataset"""
    print("ultimate reading...")
    data = pd.read_csv(data_dir + "Ultimate.txt", sep='\t', names=['type', 'ID'], header=None)
    positives = data.loc[data['type'] == 1].iloc[:, 1].values
    negatives = data.loc[data['type'] == 0].iloc[:, 1].values
    print("positives size: ", len(positives))
    print("negatives size: ", len(negatives))
    return positives, negatives

