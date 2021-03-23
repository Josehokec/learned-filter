import numpy as np
from Model import Model
from sklearn.neural_network import MLPClassifier
from utils import *


glove_dir = 'glove.6B.50d-char.txt'


def get_urldata_model():
    return MLPModel(glove_dir, 50, 50, "urldata/")


def get_shalla_model():
    return MLPModel(glove_dir, 50, 50, "shalla/")


def get_ultimate_model():
    return MLPModel(glove_dir, 50, 50, "ultimate/")


class MLPModel(Model):
    """
    Classifier: Multi-layer Perceptron(MLP) classifier
    """
    def __init__(self, embeddings_path, embedding_dim=50, maxlen=50, model_dir=""):
        """
        :param embeddings_path: glove path
        :param embedding_dim: words dimension
        :param maxlen: vector length
        :param model_dir: save mode path
        """
        # solver:{adam, sgd}, activation:{tanh, logistic, relu}, one hidden layer
        self.model = MLPClassifier(solver='adam', activation='logistic', max_iter=30,
                                   hidden_layer_sizes=(10, 5), tol=0.0005, n_iter_no_change=5)
        self.embeddings_path = embeddings_path
        self.embedding_dim = embedding_dim
        self.maxlen = maxlen
        self.model_dir = model_dir
        # others param
        self.embedding_matrix = None
        self.weights = 0
        self.bias = 0
        self.n_iter = 0

    def init_embedding_matrix(self):
        """
        Step 1: read glove file
        Step 2: convert to matrix
        """
        num_chars = len(self.char_indices)
        embedding_vectors = {}
        with open(self.embeddings_path, 'r') as f:
            for line in f:
                line_split = line.strip().split(" ")
                vec = np.array(line_split[1:self.maxlen + 1], dtype=float)
                char = line_split[0]
                embedding_vectors[char] = vec
                # embedding_vectors format: a 0.1322 0.1342 ... -0.12314

        embedding_matrix = np.zeros((num_chars + 1, self.embedding_dim), dtype=np.float)
        for char, i in self.char_indices.items():
            embedding_vector = embedding_vectors.get(char)
            assert (embedding_vector is not None)
            embedding_matrix[i] = embedding_vector
        self.embedding_matrix = embedding_matrix
        # print(self.embedding_matrix)

    def convert_format(self, int_X):
        """
        function: string index(int) array convert to float array for training
        :param int_X: int vector list, dimension is len(train_X) * maxlen
        :return: float vector list
        """
        float_X = np.zeros((len(int_X), self.maxlen), dtype=np.float)
        for i in range(len(int_X)):
            for j in range(self.maxlen):
                float_X[i, j] = self.embedding_matrix[int_X[i, j]][j]
        return float_X

    def fit(self, train_X, train_y):
        """
        Step 1: process data, string convert to int array
        Step 2: use model to fit
        :param train_X: train feature
        :param train_y: train label
        """
        # X is int array, need to convert
        X, y, self.char_indices, self.indices_char = vectorize_dataset(train_X, train_y, self.maxlen)
        self.init_embedding_matrix()
        convert_X = self.convert_format(X)
        self.model.fit(convert_X, y)
        print('model acc: ', self.model.score(convert_X, y))

        # when fit completed, save model to file

    def predict(self, test_x):
        """
        step 1: string convert to int array; step 2: use clf to predict
        :param test_x: signal item
        :return: test_x predict label is positive probability
        """
        float_x = np.zeros((1, self.maxlen), dtype=np.float)
        offset = max(self.maxlen - len(test_x), 0)
        for t, char in enumerate(test_x):
            if t >= self.maxlen:
                break
            float_x[0, t + offset] = self.embedding_matrix[self.char_indices[char]][t + offset]

        y_pred = self.model.predict_proba(float_x)

        # print('predict function testing: ', y_pred[0])
        return y_pred[0][1]

    def predicts(self, test_X):
        """
        step 1: string convert to int array; step 2: use clf to predict
        :param test_X: test item list
        :return: test_x predict label is positive probability
        """

        X = np.zeros((len(test_X), self.maxlen), dtype=np.float)
        for i in range(len(test_X)):
            offset = max(self.maxlen - len(test_X[i]), 0)
            for t, char in enumerate(test_X[i]):
                if t >= self.maxlen:
                    break
                X[i, t + offset] = self.embedding_matrix[self.char_indices[char]][t + offset]

        # pre[0] is predict label is negative probability
        # pre[1] is predict label is positive probability
        preds = [pred[1] for pred in self.model.predict_proba(X)]
        return preds

    def get_features(self, string_X):
        """
        function: Look at the eigenvector of a string
        :param string_X: strings
        :return: eigenvectors
        """
        int_X = np.zeros((len(string_X), self.maxlen), dtype=np.int)
        float_X = np.zeros((len(string_X), self.maxlen), dtype=np.float)
        for i, url in enumerate(string_X):
            offset = max(self.maxlen - len(url), 0)
            for t, char in enumerate(url):
                if t >= self.maxlen:
                    break
                int_X[i, t + offset] = self.char_indices[char]
                float_X[i, t + offset] = embedding_matrix[self.char_indices[char]][t + offset]
        return int_X, float_X

    def load_model(self):
        print('load_model function does not be implemented')

    def save_model(self):
        print('save_model function does not be implemented')

