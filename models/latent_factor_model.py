from collections import defaultdict
class LatentFactorModel(object):
    def __init__(self, train_data, val_data):
        self.train_data = train_data
        self.val_data = val_data
        self.recipes = set()
        self.users = set()
        self.recipesPerUser = {}
        self.usersPerRecipe = {}
        self.alpha = 0.0
        self.betaU = defaultdict(float)
        self.betaI = defaultdict(float)
        self.ratingPerUserRecipe = defaultdict(dict)
        self.globalAverageRating = 0.0

    def preCompute(self):
        self.recipes = set()
        self.users = set()
        self.recipesPerUser = defaultdict(set)
        self.usersPerRecipe = defaultdict(set)
        for user, recipe, data in self.train_data:
            self.recipes.add(recipe)
            self.users.add(user)
            self.recipesPerUser[user].add(recipe)
            self.usersPerRecipe[recipe].add(user)
            self.ratingPerUserRecipe[user][recipe] = data['rating']
            self.globalAverageRating += data['rating']
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
                update += self.recipesPerUser[u][i] - self.alpha - self.betaI[i]
            self.betaU[u] = update / (lamb + len(self.recipesPerUser[u]))
        for g in self.usersPerRecipe:
            update = 0
            for u in self.usersPerRecipe[g]:
                update += self.recipesPerUser[u][g] - self.alpha - self.betaU[u]
            self.betaI[g] = update / (lamb + len(self.usersPerRecipe[g]))
    def train(self, lamb, iterations):
        self.preCompute()
        for i in range(iterations):
            self.iteration(lamb)


    def predict(self, user_ids, item_ids):
        predictions = []
        for i in range(len(user_ids)):
            u, r = user_ids[i], item_ids[i]
            predictions.append(self.alpha + self.betaU[u] + self.betaI[r])
        return predictions