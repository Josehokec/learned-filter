import sys
import os

sys.path.append("../lib")

from PerceptronModel import PerceptronModel, get_urldata_model, get_shalla_model
from PLBF import PLBF
from dataload import *
from utils import *
from time import *


def test_shalla(total_size=100000, model_size=int(70 * 1024 * 8)):
    positives, negatives, costs = getShalla()
    begin_time = time()
    data = Data(positives, negatives)
    model = get_shalla_model()
    plbf = PLBF(model, data, using_Fpr=True, total_size=total_size, model_size=model_size, is_train=True)
    end_time = time()
    run_time = end_time - begin_time
    print('PLBF buildtime：', run_time)

    print("negatives fpr and query latency test...")
    query_begin_time = time()
    count = 0
    # for i in tqdm(range(len(negatives)), ncols=50):
    for i in range(len(negatives)):
        if plbf.check(negatives[i]):
            count += 1
    query_end_time = time()
    sum_query_time = query_end_time - query_begin_time
    print("fpr:", 100 * count / (len(negatives)))
    # query latency -> negative
    print("query latency:", (10 ** 6) * sum_query_time / (len(negatives)))


def test_urldata(total_size=100000, model_size=int(70 * 1024 * 8)):
    positives, negatives = get_urldata()
    begin_time = time()
    data = Data(positives, negatives)
    model = get_urldata_model()
    plbf = PLBF(model, data, using_Fpr=True, total_size=total_size, model_size=model_size, is_train=True)
    end_time = time()
    run_time = end_time - begin_time
    print('PLBF buildtime：', run_time)

    print("negatives fpr and query latency test...")
    query_begin_time = time()
    count = 0
    # from tqdm import tqdm
    # for i in tqdm(range(len(negatives)), ncols=50):
    for i in range(len(negatives)):
        if plbf.check(negatives[i]):
            count += 1
    query_end_time = time()
    sum_query_time = query_end_time - query_begin_time
    print("fpr:", 100 * count / (len(negatives)))
    # query latency -> negative
    print("query latency:", (10**6) * sum_query_time / (len(negatives)))


if __name__ == '__main__':
    # url test
    test_urldata()
