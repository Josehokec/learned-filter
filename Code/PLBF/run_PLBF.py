import sys
import os
import json
import tensorflow as tf

sys.path.append("../lib")

from GRUModel import GRUModel, get_urldata_model
from PLBF import PLBF
from dataload import *
from utils import *
from time import *
from tqdm import tqdm


def test_urldata(total_size=100000, model_size=int(70 * 1024 * 8)):
    positives, negatives = get_urldata()
    begin_time = time()
    data = Data(positives, negatives)
    model = get_urldata_model()
    plbf = PLBF(model, data, using_Fpr=True, total_size=total_size, model_size=model_size, is_train=True)
    end_time = time()
    run_time = end_time - begin_time
    print('LBF buildtime：', run_time)

    print("negatives testing")      # fpr and query latency test
    query_begin_time = time()
    count = 0
    for i in tqdm(range(len(negatives)), ncols=50):
        if plbf.check(negatives[i]):
            count += 1
    query_end_time = time()
    sum_query_time = query_end_time - query_begin_time
    print("fpr:", 100 * count / (len(negatives)))
    # query latency -> negative
    print("query latency:", (10**6) * sum_query_time / (len(negatives)))


if __name__ == '__main__':
    GPUFLAG = True          # GPU config
    if GPUFLAG:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

    # url test
    test_urldata()
