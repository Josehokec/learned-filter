import sys
import os
import json

sys.path.append("../lib")

from dataload import *
from utils import *
from time import *
from tqdm import tqdm
from BloomFilter import BestBloomFilter

def test_shalla_cost(total_size=100000, model_size=int(70 * 1024 * 8)):
    positives, negatives, neg_cost = getShalla()
    begin_time = time()
    data = Data(positives, negatives)
    model = get_shalla_model()
    lbf = LBF(model, data, using_Fpr=False, total_size=total_size, model_size=model_size, is_train=True)
    end_time = time()
    run_time = end_time - begin_time
    print('LBF buildtime：', run_time)

    # cost test
    print("cost testing")
    count = 0
    for i in tqdm(range(len(negatives)), ncols=50):
        if lbf.check(negatives[i]):
            count += 1
    print("fpr:", 100 * count / (len(negatives)))


def test_urldata(fpr=0.01):
    positives, negatives = get_urldata()
    begin_time = time()
    data = Data(positives, negatives)
    bf = BestBloomFilter(len(positives), fpr)
    for pos_key in positives:
        bf.add(pos_key)
    end_time = time()
    run_time = end_time - begin_time
    print('BF buildtime: ', run_time)
    print('size of memory: ', bf.size)
    print('number of hash: ', bf.hash_count)

    # fpr & query latency
    query_begin_time = time()
    count = 0
    for i in tqdm(range(len(negatives)), ncols=50):
        if bf.check(negatives[i]):
            count += 1
    query_end_time = time()
    sum_query_time = query_end_time - query_begin_time
    # fpr: %    query latency: ns
    print("fpr:", 100 * count / (len(negatives)))
    print("query latency:", (10**6) * sum_query_time / (len(negatives)))


if __name__ == '__main__':
    test_urldata()
