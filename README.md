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

3. To train the model, download the dataset from [here](https://www.kaggle.com/shuyangli94/food-com-recipes-and-user-interactions)  to the `data/` folder and run the following command
```sh
python train.py --model <model> --topk <topk> ---lr <lr> --lambd <lambd> --n_estimators <n_estimators> --max_depth <max_depth>
```

