
from sklearn.ensemble import RandomForestRegressor
import os
from pathlib import Path
import sys
path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))
from commons.constants import *
from commons.utils import *

class RandomForestModel:
    def __init__(self, n_estimators=100, topk=12, max_depth=10):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.topk = topk
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth)

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