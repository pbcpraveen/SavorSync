import torch
import torch.nn as nn
import torch.optim as optim
import sys
import os
from pathlib import Path
path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))
from commons.constants import *
from commons.utils import *

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, input_dim, lr=0.001, n_iters=1000):
        super(SimpleNeuralNetwork, self).__init__()
        self.layer1 = nn.Linear(input_dim, 16)
        self.layer2 = nn.Linear(16, 8)
        self.output_layer = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = self.output_layer(x)
        return x

    def train(self, data, epochs=50, learning_rate=0.001):
        data = getProcessedData('train')
        X_train = data[0]
        y_train = data[1]
        pairs = data[2]

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            inputs = torch.tensor(X_train, dtype=torch.float32)
            labels = torch.tensor(y_train, dtype=torch.float32)

            optimizer.zero_grad()
            outputs = self.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')
