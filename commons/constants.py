from enum import Enum
import os
from pathlib import Path
import sys
path = Path(os.getcwd())
sys.path.append(str(path.parent.absolute()))

class RecipeUserFeature(Enum):
    TIME_TAKEN = 'timeTaken'
    NUM_INGREDIENTS = 'numIngredients'
    NUMBER_OF_STEPS = 'numberOfSteps'
    AVG_RATING_FOR_USER = 'avgRatingForUser'
    AVG_RATING_FOR_RECIPE = 'avgRatingForRecipe'
    CALORIES = 'calories'
    TOTAL_FAT = 'totalFat'
    SUGAR = 'sugar'
    SODIUM = 'sodium'
    PROTEIN = 'protein'
    SATURATED_FAT = 'saturatedFat'
    CARBOHYDRATES = 'carbohydrates'
    RATING = 'rating'

# set path from parent directory data/archive/interaction_text.csv
path = Path(os.getcwd())
parent = path.parent.absolute()
parent = parent / 'SavourSync'
normalising = parent / 'data_analysis/data/normalising_factor.pkl'

test_path = parent / 'data/archive/interactions_test.csv'
train_path = parent / 'data/archive/interactions_train.csv'
val_path = parent / 'data/archive/interactions_validation.csv'

processed_train_path = parent / 'data_analysis/data/trainData.pkl'
processed_test_path = parent / 'data_analysis/data/testData.pkl'
processed_val_path = parent / 'data_analysis/data/valData.pkl'

raw_interaction = parent / 'data/archive/RAW_interactions.csv'
raw_recipe = parent / 'data/archive/RAW_recipes.csv'

data_path = {
    'test': test_path,
    'train': train_path,
    'val': val_path
}

processed_data_path = {
    'test': processed_test_path,
    'train': processed_train_path,
    'val': processed_val_path
}

