import pandas as pd

def load_data():
    dataset = pd.read_csv('data/Womens Clothing E-Commerce Reviews.csv')
    dataset = dataset[0:100]
    return dataset
