import math
import random
import numpy as np
from BloomFilter import BestBloomFilter, BloomFilter
from utils import *
from time import *


class MLPLBF(object):
    """
    Practical learned bloom filter use Multi Layers perceptron as classifier
    """
    def __init__(self, model, data, using_Fpr=True, fp_rate=0.01, total_size=100000, model_size=int(70 * 1024 * 8),
                 is_train=True):
        self.model = model
        self.threshold = 0.9
        self.using_Fpr = using_Fpr
        self.is_train = is_train

        # divide ratio = 0.7
        (s1, s2) = split_negatives(data, 0.7)

        # whether need to train model, if no, load trained model
        if self.is_train:
            self.fit(data.positives, data.negatives)
        else:
            self.model.load_model()

        # whether need to use fpr, if yes, then according to fpr create BF
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
        fit_begin_time = time()         # ->test train time
        self.model.fit(shuffled[0], shuffled[1])
        fit_end_time = time()           # ->test train time
        fit_time = fit_end_time - fit_begin_time
        print('fitting spend time:', fit_time)
        print("Done fitting")
        """
        neg_scores = self.model.predicts(negatives)
        pos_scores = self.model.predicts(positives)
        show_distribute(pos_scores, neg_scores, "")
        exit()
        
        # this is a test for model
        self.test_model(positives, negatives)
        exit()
        """

    def get_threshold(self, test_negatives):
        fp_index = math.ceil((len(test_negatives) * (1 - self.fp_rate / 2)))
        predictions = self.model.predicts(test_negatives)
        predictions.sort()
        # set threshold
        self.threshold = predictions[fp_index]

    def test_model(self, positives, negatives):
        print(negatives[0] + 'proba is: ')
        self.model.predict(negatives[0])
        len_test = 15
        neg_scores = self.model.predicts(negatives[0:len_test])
        pos_scores = self.model.predicts(positives[0:len_test])
        for i in range(len_test):
            print(positives[i] + ' proba is(P): ' + str(pos_scores[i]))
            print(negatives[i] + ' proba is(N): ' + str(neg_scores[i]))
