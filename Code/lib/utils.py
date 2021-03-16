import random
import mmh3
import numpy as np
from collections import Counter


class Data(object):
    def __init__(self, positives, negatives):
        self.positives = positives
        self.negatives = negatives


def shuffle_for_training(negatives, positives):
    a = [(i, 0) for i in negatives]
    b = [(i, 1) for i in positives]
    combined = a + b
    random.shuffle(combined)
    return list(zip(*combined))


def compact1(negatives, positives):
    a = [(i, 0) for i in negatives]
    b = [(i, 1) for i in positives]
    combined = a + b
    return list(zip(*combined))


def string_digest(item, index):
    return mmh3.hash(bytes(item, 'utf-8'), index)


def split_negatives(data, train_portion=0.9):
    size = len(data.negatives)
    s1 = data.negatives[0:int(train_portion * size)]
    s2 = data.negatives[int(train_portion * size):]
    return (s1, s2)


def split_positives(data, store_portion=0.4):
    """
    Initial bloom filter stores part of keys
    s1: store in initial fiter
    s2: training for learned oracle
    """
    size = len(data.positives)
    s1 = data.positives[0:int(store_portion * size)]
    s2 = data.positives[int(store_portion * size):]
    return (s1, s2)


def vectorize_dataset(text_X, text_y, maxlen):
    # Adapted from Keras examples
    print("Vectorizing data...")
    raw_text = ''.join(text_X)
    print("Corpus length", len(raw_text))
    chars = sorted(list(set(raw_text)))
    print(chars)
    print('Total chars:', len(chars))

    lengths = [len(url) for url in text_X]
    counter = Counter(lengths)
    counts = sorted([(key, counter[key]) for key in counter])
    # print(counts)
    max_seen = 0
    for url in text_X:
        max_seen = max(len(url), max_seen)
    print("max seen length of URL", max_seen)
    print("Using maxlen", maxlen)
    char_indices = dict((c, i + 1) for i, c in enumerate(chars))
    indices_char = dict((i + 1, c) for i, c in enumerate(chars))

    # 0 in this indicates empty word, 1 through len(chars) inclusive
    # indicates a particular char
    X = np.zeros((len(text_X), maxlen), dtype=np.int)
    y = np.zeros((len(text_X)), dtype=np.bool)
    for i, url in enumerate(text_X):
        offset = max(maxlen - len(url), 0)
        for t, char in enumerate(url):
            if t >= maxlen:
                break
            X[i, t + offset] = char_indices[char]
        # print(X[i])
        y[i] = 1 if text_y[i] == 1 else 0

    return X, y, char_indices, indices_char


def test_model(model, text_X, text_y):
    total = float(len(text_X))
    total_correct = 0.0
    false_positives = 0.0
    false_negatives = 0.0
    for i, url in enumerate(text_X):
        raw_pred = model.predict(url)
        pred = 1 if raw_pred > 0.5 else 0
        label = text_y[i]
        if pred == label:
            total_correct += 1
        else:
            if pred == 1:
                false_positives += 1
            else:
                false_negatives += 1
    return total_correct / total, false_positives / total, false_negatives / total


def evaluate_model(model, positives, negatives_train, negatives_dev, negatives_test, threshold):
    false_negatives = 0.0
    preds = model.predicts(positives)
    for pred in preds:
        if pred <= threshold:
            false_negatives += 1

    print(false_negatives / len(positives), "false negatives for positives set.")

    false_positives_train = 0.0
    preds = model.predicts(negatives_train)
    for pred in preds:
        if pred > threshold:
            false_positives_train += 1

    false_positives_dev = 0.0
    preds = model.predicts(negatives_dev)
    for pred in preds:
        if pred > threshold:
            false_positives_dev += 1

    false_positives_test = 0.0
    preds = model.predicts(negatives_test)
    for pred in preds:
        if pred > threshold:
            false_positives_test += 1

    print(false_positives_train / len(negatives_train), "false positive rate for train.")
    print(false_positives_dev / len(negatives_dev), "false positive rate for dev.")
    print(false_positives_test / len(negatives_test), "false positive rate for test.")
