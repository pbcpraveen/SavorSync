from sklearn.linear_model import SGDClassifier
import os
from pathlib import Path
import sys
path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))
from commons.constants import *
from commons.utils import *

class LogisticRegressionModel:
    def __init__(self, lr=0.001, n_iters=1000, topk=12):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.mean = None
        self.topk = topk
        self.std = None
        # model should support multiclass classification
        self.model = SGDClassifier(loss='log', max_iter=n_iters, eta0=self.lr, class_weight='balanced')

    def f1_score(self, y_true, y_pred):
        tp = np.sum(y_true * y_pred)
        fp = np.sum((1 - y_true) * y_pred)
        fn = np.sum(y_true * (1 - y_pred))
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        return f1

    def recall_at_2(self, y_true, y_pred):
        tp = np.sum(y_true * y_pred)
        fn = np.sum(y_true * (1 - y_pred))
        recall = tp / (tp + fn)
        return recall

    def train(self):
        data = getProcessedData('train', top_k=self.topk)
        x = data[0]
        y = data[1] * 5
        pairs = data[2]
        print(x.shape, y.shape)
        self.model.fit(x, y)
        # calculate training loss
        # get validation data
        data = getProcessedData('val', top_k=self.topk)
        x_val = data[0]
        y_val = data[1] * 5
        pairs_val = data[2]

        # calculate validation loss
        y_predicted = self.model.predict(x_val)
        val_accuracy = np.mean(y_val == y_predicted)
        val_f1 = {}
        for i in np.unique(y_val):
            val_f1[i] = self.f1_score(y_val == i, y_predicted == i)

        recall_at_2 = {}
        for i in np.unique(y_val):
            y_true = y_val == i
            log_probs = self.model.predict_proba(x_val)
            #find predictions with top 2 probabilities
            top_2 = np.argsort(log_probs, axis=1)[:,-2:]
            #find the number of times the true label is in the top 2
            recall_at_2[i] = sum([i in top_2[j] for j in range(len(top_2))]) /len(top_2)
        return val_accuracy, val_f1, recall_at_2

    def predict(self):
        data = getProcessedData('test', top_k=self.topk)
        x_test = data[0]
        y_test = data[1] * 5
        pairs_test = data[2]
        y_predicted = self.model.predict(x_test)
        test_accuracy = np.mean(y_test == y_predicted)
        test_f1 = {}
        for i in np.unique(y_test):
            test_f1[i] = self.f1_score(y_test == i, y_predicted == i)

        recall_at_2 = {}
        for i in np.unique(y_test):
            y_true = y_test == i
            log_probs = self.model.predict_proba(x_test)
            #find predictions with top 2 probabilities
            top_2 = np.argsort(log_probs, axis=1)[:,-2:]
            #find the number of times the true label is in the top 2
            recall_at_2[i] = np.sum(top_2 == i, axis=1) / 2
        return test_accuracy, test_f1, recall_at_2