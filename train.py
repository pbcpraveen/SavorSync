
import sys
import os
from pathlib import Path
path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))
from commons.constants import *
from commons.utils import *
from models.linear_regression import LinearRegressionModel
from models.neural_network import SimpleNeuralNetwork
from models.latent_factor_model import LatentFactorModel
import argparse

def main(model):
    # linear regression
    train_mse, val_mse = model.train()
    print(f'Train MSE: {train_mse}, Validation MSE: {val_mse}')
    test_mse = model.predict()
    print(f'Test MSE: {test_mse}')

if __name__ == '__main__':
    #create command line arguments for topk, learning rate, model, lambda
    parser = argparse.ArgumentParser()
    parser.add_argument('--topk', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--model', type=str, default='linear')
    parser.add_argument('--lambd', type=float, default=0.001)
    args = parser.parse_args()
    print("====================================================")
    print("Running with arguments: ", args)
    if args.model == 'linear':
        model = LinearRegressionModel(lr=args.lr, topk=args.topk)
    elif args.model == 'nn':
        model = SimpleNeuralNetwork(input_dim=args.topk, lr=args.lr)
    elif args.model == 'lfm':
        model = LatentFactorModel(args.lambd)
    else:
        raise ValueError('Model not supported')
    main(model)
    print("====================================================")