import sys
import pickle
import math
import random

from BloomFilter import BestBloomFilter, BloomFilter
from utils import *

sys.path.append("../lib")

"""
add a initial filter
"""
class PLBF(object):
    def __init__(self, model, data, using_Fpr=True, fp_rate=0.01, total_size=100000, model_size=int(70 * 1024 * 8),
                 is_train=True):
        self.model = model
        self.threshold = 0.9
        self.using_Fpr = using_Fpr
        self.is_train = is_train
        
        (s1, s2) = split_positives(data, 0.3)   # s1 are stored in Inital Filter, s2 are used for training
        (s3, s4) = split_negatives(data)        # s4 are used determine value of threshold
        data.positives = s2                     # update dataset
        
        print('Create initial filter...')       # create initial filter to store s1
        self.initial_bf = BestBloomFilter(len(s1), fp_rate / 2)
        for key in s1:
            self.initial_bf.add(key)
        print('size of memory in initial_bf:', self.initial_bf.size)
        print('number of K in initial_bf:', self.initial_bf.hash_count)
        
        if self.is_train:
            self.fit(data.positives, data.negatives)
        else:
            self.model.loadmodel()
            print("load model success!!!")

        if using_Fpr:
            # default use using fpr
            self.fp_rate = float(fp_rate)
            self.create_best_bloom_filter(data, s4)
        else:
            self.m = total_size - model_size
            self.create_bloom_filter(data, s4)

    def check(self, item):
        if self.initial_bf.check(item):
            return True
        elif self.model.predict(item) > self.threshold:
            return True
        return self.bloom_filter.check(item)


    def create_best_bloom_filter(self, data, test_negatives):
        print("Creating bloom filter")
        self.get_threshold(test_negatives)
        print("threshold: %f" % self.threshold)
        
        false_negatives = []
        preds = self.model.predicts(data.positives)
        print("Number of positive key", len(preds))
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

        self.threshold = predictions[fp_index]
