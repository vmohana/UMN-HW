import pandas as pd
import math
#from sklearn import tree

# Preprocess the dataset
dataset = pd.read_csv('dataset.csv', header=None)
dataset.columns = ['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est', 'Wait']
dataset[dataset.columns] = dataset.apply(lambda x: x.str.strip())

def pure(dataset):
    if len(dataset['Wait'].unique()) == 1:
        return True
    else:
        return False

def split_set(dataset, attribute):
    split = {}
    for vk in dataset[attribute].unique():
        split[vk] = dataset[dataset[attribute] == vk]
    return split

def decision_list(examples):
    d_list = []
    while True:
        for attribute in examples.columns[:-1]:
            split_data = split_set(examples, attribute)
            for vk in list(split_data.keys()):
                if pure(split_data[vk]):
                    d_list.append((attribute, vk, split_data[vk]['Wait'].iloc[0]))
                    examples = examples.drop(examples[examples[attribute] == vk].index)
                if len(examples) == 0:
                    return d_list
    return d_list
