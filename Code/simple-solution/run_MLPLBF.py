from dataload import *
from utils import *
from time import *
import warnings
from MLPLBF import MLPLBF
from MLPModel import MLPModel, get_urldata_model, get_shalla_model, get_ultimate_model


def test_shalla(total_size=100000, model_size=int(70 * 1024 * 8)):
    positives, negatives, costs = getShalla()
    begin_time = time()
    data = Data(positives, negatives)
    data.random_shuffle()
    model = get_shalla_model()
    plbf = MLPLBF(model, data, using_Fpr=True, total_size=total_size, model_size=model_size, is_train=True)
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
    data.random_shuffle()
    model = get_urldata_model()
    plbf = MLPLBF(model, data, using_Fpr=True, total_size=total_size, model_size=model_size, is_train=True)
    end_time = time()
    run_time = end_time - begin_time
    print('PLBF buildtime：', run_time)

    """
    pos_scores = model.predicts(data.positives)
    neg_scores = model.predicts(data.negatives)
    show_distribute(pos_scores, neg_scores, 'urldata')
    """

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


def test_ultimate(total_size=100000, model_size=int(70 * 1024 * 8)):
    positives, negatives = get_ultimate()
    begin_time = time()
    data = Data(positives, negatives)
    data.random_shuffle()
    model = get_urldata_model()
    plbf = MLPLBF(model, data, using_Fpr=True, total_size=total_size, model_size=model_size, is_train=True)
    end_time = time()
    run_time = end_time - begin_time
    print('PLBF buildtime：', run_time)


    pos_scores = model.predicts(data.positives)
    neg_scores = model.predicts(data.negatives)
    show_distribute(pos_scores, neg_scores, 'ultimate')


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
    warnings.filterwarnings("ignore")
    test_ultimate()
