import sys
import pickle
import math
import random

from BloomFilter import BestBloomFilter, BloomFilter
from utils import *

sys.path.append("../lib")


class PLBF(object):
    """
    Practical learned bloom filter use perceptron as classifier
    """
    def __init__(self, model, data, using_Fpr=True, fp_rate=0.01, total_size=100000, model_size=int(70 * 1024 * 8),
                 is_train=True):
        self.model = model
        self.threshold = 0.9
        self.using_Fpr = using_Fpr
        self.is_train = is_train
        (s1, s2) = split_negatives(data, 0.7)
        if self.is_train:
            self.fit(data.positives, data.negatives)
        else:
            self.model.load_model()
            print("model load success")
        if using_Fpr:
            self.fp_rate = float(fp_rate)
            self.create_best_bloom_filter(data, s2)
        else:
            self.m = total_size - model_size
            self.create_bloom_filter(data, s2)

    def check(self, item):
        if self.model.predict(item) > self.threshold:
            return True
        return self.bloom_filter.check(item)

    def create_best_bloom_filter(self, data, test_negatives):
        print("Creating bloom filter")
        self.get_threshold(test_negatives)
        print("model threshold: %f" % self.threshold)

        false_negatives = []
        preds = self.model.predicts(data.positives)
        for i in range(len(data.positives)):
            if preds[i] <= self.threshold:
                false_negatives.append(data.positives[i])
        print("Number of false negatives at bloom time", len(false_negatives))
        self.bloom_filter = BestBloomFilter(len(false_negatives), self.fp_rate / 2)
        for fn in false_negatives:
            self.bloom_filter.add(fn)
        print("Created bloom filter")
        print("hash function K: ", self.bloom_filter.hash_count)
        print("bBF memory size: ", self.bloom_filter.size)

    def fit(self, positives, negatives):
        shuffled = shuffle_for_training(negatives, positives)
        self.model.fit(shuffled[0], shuffled[1])
        print("Done fitting")

    def get_threshold(self, test_negatives):
        fp_index = math.ceil((len(test_negatives) * (1 - self.fp_rate / 2)))
        predictions = self.model.predicts(test_negatives)
        predictions.sort()

        """
        import pandas as pd
        excel_data = pd.DataFrame(predictions)
        writer = pd.ExcelWriter('preds.xlsx')  # 写入Excel文件
        excel_data.to_excel(writer, float_format='%.5f')
        writer.save()
        writer.close()
        # ---------------------------------------------
        predictions = self.model.predicts(test_negatives[0:10])
        print(predictions)
        result1 = list()
        result2 = list()
        for i in range(10):
            result1.append(self.model.predict(test_negatives[i]))
        for i in range(10):
            result2.append(self.model.predict(data.positives[i]))
        print('negative keys test: ', result1)
        print('positive keys test: ', result2)
        """

        self.threshold = predictions[fp_index]
