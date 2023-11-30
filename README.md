# SavorSync: A Culinary Compass for Personalized Recipe Journeys
### A recipe recommendation system based on user preferences and interactions.

## Get Started
### Prerequisites
- Python 3.7 or higher
- pip
### Installation
1. Clone the repo
```sh
git clone https://github.com/pbcpraveen/SavourSync
```
2. Install Python packages
```sh
pip install -r requirements.txt
```
3. Run inference for a user and recipe pair
```sh
python inference.py --user_id <user_id> --recipe_id <recipe_id>
```
4. To train the model, download the dataset from [here](https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions)  to the `data/` folder and run the following command
```sh
python train.py --data_path <path_to_dataset>
```

