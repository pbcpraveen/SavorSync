from collections import defaultdict
import os
from pathlib import Path
import sys

path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))
from commons.constants import *
from commons.utils import *


class LatentFactorModel(object):
    def __init__(self, lamb, iterations):
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.recipes = set()
        self.users = set()
        self.recipesPerUser = {}
        self.usersPerRecipe = {}
        self.alpha = 0.0
        self.betaU = defaultdict(float)
        self.betaI = defaultdict(float)
        self.ratingPerUserRecipe = defaultdict(dict)
        self.globalAverageRating = 0.0
        self.lamb = lamb
        self.iterations = iterations
        self.train_loss = []
        self.val_loss = []

    def preCompute(self):
        x, y, pairs = getProcessedData('train', top_k=12)
        self.train_data = [list(a1) + [b1] for (a1, b1) in zip(pairs, y)]
        self.recipes = set()
        self.users = set()
        self.recipesPerUser = defaultdict(set)
        self.usersPerRecipe = defaultdict(set)
        for user, recipe, data in self.train_data:
            self.recipes.add(recipe)
            self.users.add(user)
            self.recipesPerUser[user].add(recipe)
            self.usersPerRecipe[recipe].add(user)
            self.ratingPerUserRecipe[user][recipe] = data
            self.globalAverageRating += data
        self.globalAverageRating /= len(self.train_data)
        self.alpha = sum([r for _, _, r in self.train_data]) / len(self.train_data)
        self.betaU = defaultdict(float)
        self.betaI = defaultdict(float)
        for user in self.users:
            self.betaU[user] = 0.0
        for recipe in self.recipes:
            self.betaI[recipe] = 0.0

    def iteration(self, lamb):
        self.alpha = self.globalAverageRating
        for i in range(len(self.train_data)):
            u, r = self.train_data[i][0], self.train_data[i][1]
            self.alpha += self.ratingPerUserRecipe[u][r] - self.betaU[u] - self.betaI[r]
        self.alpha /= len(self.train_data)
        for u in self.recipesPerUser:
            update = 0
            for i in self.recipesPerUser[u]:
                update += self.ratingPerUserRecipe[u][i] - self.alpha - self.betaI[i]
            self.betaU[u] = update / (lamb + len(self.recipesPerUser[u]))
        for g in self.usersPerRecipe:
            update = 0
            for u in self.usersPerRecipe[g]:
                update += self.ratingPerUserRecipe[u][g] - self.alpha - self.betaU[u]
            self.betaI[g] = update / (lamb + len(self.usersPerRecipe[g]))
        pred = []
        for i in range(len(self.train_data)):
            u, r = self.train_data[i][0], self.train_data[i][1]
            pred.append(self.alpha + self.betaU[u] + self.betaI[r])
        mse = np.mean((np.array(pred) - np.array([i[2] for i in self.train_data])) ** 2)
        self.train_loss.append(mse)

    def train(self):
        self.preCompute()
        x, y, pairs = getProcessedData('val', top_k=12)
        self.val_data = pairs
        for i in tqdm(range(self.iterations)):
            self.iteration(self.lamb)
            y_pred = []
            for i in range(len(self.val_data)):
                u, r = self.val_data[i][0], self.val_data[i][1]
                y_pred.append(self.alpha + self.betaU[u] + self.betaI[r])
            mse = np.mean((np.array(y_pred) - np.array(y)) ** 2)
            self.val_loss.append(mse)



        val_predictions = []
        for i in range(len(self.val_data)):
            u, r = self.val_data[i][0], self.val_data[i][1]
            val_predictions.append(self.alpha + self.betaU[u] + self.betaI[r])

        mse = np.mean((np.array(val_predictions) - np.array(y)) ** 2)
        print("MSE on validation set: ", mse)

        return 0, mse

    def predict(self):
        predictions = []
        x, y, pairs = getProcessedData('test', top_k=12)
        self.test_data = pairs
        for i in range(len(self.test_data)):
            u, r = self.test_data[i][0], self.test_data[i][1]
            predictions.append(self.alpha + self.betaU[u] + self.betaI[r])

        mse = np.mean((np.array(predictions) - np.array(y)) ** 2)
        return mse
