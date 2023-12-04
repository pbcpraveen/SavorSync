from sklearn.linear_model import SGDRegressor
import os
from pathlib import Path
import sys
path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))
from commons.constants import *
from commons.utils import *

class LinearRegressionModel:
    def __init__(self, lr=0.001, n_iters=1000, topk=12):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.mean = None
        self.topk = topk
        self.std = None
        self.model = SGDRegressor(max_iter=n_iters, eta0=self.lr,)

    def train(self):
        data = getProcessedData('train', top_k=self.topk)
        x = data[0]
        y = data[1]
        pairs = data[2]
        print(x.shape, y.shape)
        self.model.fit(x, y)
        # calculate training loss
        y_predicted = self.model.predict(x)
        train_mse = np.mean((y - y_predicted)**2)
        # get validation data
        data = getProcessedData('val', top_k=self.topk)
        x_val = data[0]
        y_val = data[1]
        pairs_val = data[2]

        # calculate validation loss
        y_predicted = self.model.predict(x_val)
        val_mse = np.mean((y_val - y_predicted)**2)
        return train_mse, val_mse



    def predict(self):
        data = getProcessedData('test', top_k=self.topk)
        x_test = data[0]
        y_test = data[1]
        pairs_test = data[2]
        y_predicted = self.model.predict(x_test)
        test_mse = np.mean((y_test - y_predicted)**2)
        return test_mse
