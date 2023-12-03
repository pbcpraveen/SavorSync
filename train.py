
import sys
import os
from pathlib import Path
path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))
from commons.constants import *
from commons.utils import *
from models.linear_regression import LinearRegressionModel
from models.neural_network import SimpleNeuralNetwork

def main():
    # linear regression
    model = LinearRegressionModel()
    train_mse, val_mse = model.train()
    print(f'Train MSE: {train_mse}, Validation MSE: {val_mse}')
    test_mse = model.predict()
    print(f'Test MSE: {test_mse}')

if __name__ == '__main__':
    main()