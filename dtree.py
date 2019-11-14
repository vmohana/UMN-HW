import pandas as pd
import math
from sklearn import tree

# Preprocess the dataset
dataset = pd.read_csv('dataset.csv', header=None)
dataset.columns = ['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est', 'Wait']
dataset[dataset.columns] = dataset.apply(lambda x: x.str.strip())

def entropy(dataset):
    unique_values = dataset['Wait'].unique()
    entropy = 0
    for v in unique_values:
        entropy += (len(dataset[dataset['Wait']==v])/len(dataset))*math.log2(len(dataset[dataset['Wait']==v])/len(dataset))
    return -1*entropy

def get_split_attribute(dataset):

    # Calculate dataset entropy
    unique_target = dataset['Wait'].unique()
    dataset_entropy = 0
    for v in unique_target:
        dataset_entropy += (len(dataset[dataset['Wait']==v])/len(dataset))*math.log2(len(dataset[dataset['Wait']==v])/len(dataset))
    dataset_entropy *= -1

    # Calculate entropy of each attribute
    attributes = dataset.columns[:-1]
    best_split = None
    attribute_entropies = {}
    for attribute in attributes:
        attribute_entropy = 0
        unique_values = dataset[attribute].unique()
        for unique_value in unique_values:
            sub_dataset = dataset[dataset[attribute] == unique_value]
            unique_value_probability = len(sub_dataset)/len(dataset)
            attribute_entropy += unique_value_probability*entropy(sub_dataset)
        attribute_entropies[attribute] = dataset_entropy - attribute_entropy
    return sorted(attribute_entropies.items(), key = lambda kv: kv[1])[-1][0]

def split_dataset(attribute, dataset):
    split_data = {}
    for unique_value in dataset[attribute].unique():
        #print(unique_value)
        split_data[unique_value] = dataset[dataset[attribute] == unique_value]
        del split_data[unique_value][attribute]
    return split_data


def decision_tree(dataset):
    split_attribute = get_split_attribute(dataset)
    split_data = split_dataset(split_attribute, dataset)
    tree = {}
    tree[split_attribute] = {}
    for attribute_value in dataset[split_attribute].unique():
        if entropy(split_data[attribute_value]) == 0:
            tree[split_attribute][attribute_value] = split_data[attribute_value]['Wait'].iloc[0]
        else:
            tree[split_attribute][attribute_value] = decision_tree(split_data[attribute_value])
    return tree
