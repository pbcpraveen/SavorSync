import os
from pathlib import Path
import sys
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import pickle
import numpy as np

from commons.constants import normalising

path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))
from commons.constants import *


def process_data(partition="train"):
    datapath = data_path[partition]
    data = pd.read_csv(datapath)
    raw_recipes = pd.read_csv(raw_recipe)
    lookup = defaultdict(dict)
    for i in tqdm(range(len(raw_recipes))):
        recipe_id = raw_recipes.iloc[i]['id']
        lookup[recipe_id][RecipeUserFeature.TIME_TAKEN.value] = int(raw_recipes.iloc[i]['minutes'])
        lookup[recipe_id][RecipeUserFeature.NUM_INGREDIENTS] = len(raw_recipes.iloc[i]['ingredients'].split(','))
        lookup[recipe_id]['nutrition'] = raw_recipes.iloc[i]['nutrition']
        lookup[recipe_id][RecipeUserFeature.NUMBER_OF_STEPS.value] = int(raw_recipes.iloc[i]['n_steps'])

    interactions_train = pd.read_csv(data_path['train'])

    ratingsPerUser = defaultdict(list)
    for index, row in tqdm(interactions_train.iterrows()):
        ratingsPerUser[row['user_id']].append(row['rating'])

    avgRatingPerUser = defaultdict(float)
    for user_id, ratings in ratingsPerUser.items():
        avgRatingPerUser[user_id] = np.mean(ratings)

    ratingsPerRecipe = defaultdict(list)
    for index, row in tqdm(interactions_train.iterrows()):
        ratingsPerRecipe[row['recipe_id']].append(row['rating'])

    avgRatingPerRecipe = defaultdict(float)
    for recipe_id, ratings in ratingsPerRecipe.items():
        avgRatingPerRecipe[recipe_id] = np.mean(ratings)

    recipe_features = [
        RecipeUserFeature.TIME_TAKEN.value,
        RecipeUserFeature.NUM_INGREDIENTS,
        'nutrition',
        RecipeUserFeature.NUMBER_OF_STEPS.value]

    x = []
    y = []
    pairs = []
    averageTimeTaken = raw_recipes['minutes'].mean()
    averageNumIngredients = raw_recipes['ingredients'].apply(lambda x: len(x.split(','))).mean()
    averageNumberOfSteps = raw_recipes['n_steps'].mean()
    globalAverageRating = interactions_train['rating'].mean()
    averageNutrition = np.array([eval(lookup[recipe_id]['nutrition']) for recipe_id in lookup]).mean(axis=0)

    for i in tqdm(range(len(data))):
        user_id = data.iloc[i]['u']
        recipe_id = data.iloc[i]['i']
        rating = data.iloc[i]['rating']
        if recipe_id in avgRatingPerRecipe:
            timeTaken = lookup[recipe_id][RecipeUserFeature.TIME_TAKEN.value]
            numIngredients = lookup[recipe_id][RecipeUserFeature.NUM_INGREDIENTS]
            nutrition = eval(lookup[recipe_id]['nutrition'])
            numberOfSteps = lookup[recipe_id][RecipeUserFeature.NUMBER_OF_STEPS.value]
            avgRatingForRecipe = avgRatingPerRecipe[recipe_id]
        else:
            timeTaken = averageTimeTaken
            numIngredients = averageNumIngredients
            nutrition = averageNutrition
            numberOfSteps = averageNumberOfSteps
            avgRatingForRecipe = globalAverageRating
        if user_id in avgRatingPerUser:
            avgRatingForUser = avgRatingPerUser[user_id]
        else:
            avgRatingForUser = globalAverageRating
        data = [timeTaken, numIngredients, numberOfSteps, avgRatingForUser, avgRatingForRecipe]
        data.extend(nutrition)
        label = rating
        x.append(data)
        y.append(label)
        pairs.append((user_id, recipe_id))

        return x, y, pairs

def getProcessedData(partition="train"):
    """Get the preprocessed data from the pickle file."""
    path = Path(os.getcwd())
    parent = path.parent.absolute()
    datapath = data_path[partition]
    if not os.path.exists(datapath):
        raise Exception(f"Path {datapath} does not exist.")
    processed_data = processed_data_path[partition]
    if not os.path.exists(processed_data):
        x, y, pairs = process_data(partition)
        x = np.array(x)
        y = np.array(y)
        mean_train = None
        if partition == 'train':
            mean_train = np.mean(x, axis=0)
            std_train = np.std(x, axis=0)
            normalising_factor = (mean_train, std_train)
            with open(normalising, 'wb') as f:
                pickle.dump(normalising_factor, f)
        else:
            if not os.path.exists(normalising):
                raise Exception(f"Path {normalising} does not exist.")
            with open(normalising, 'rb') as f:
                normalising_factor = pickle.load(f)
            mean_train, std_train = normalising_factor
        x = (x - mean_train) / std_train
        y = y / 5.0
        with open(processed_data, 'wb') as f:
            pickle.dump((x, y, pairs), f)
    with open(processed_data_path[partition], 'rb') as f:
        data = pickle.load(f)
    return data
