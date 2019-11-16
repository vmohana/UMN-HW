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


'''
Recursive
def decision_list(dataset):
    if len(dataset) == 0:
        return ('T', 'F')
    else:
        split_attribute = None
        split_value = None
        for attribute in dataset.columns[0:10]:
            sub_data = split_data(dataset, attribute)
            for value in list(sub_data.keys()):
                if pure(sub_data[value]):
                    split_attribute = attribute
                    split_value = value
                    break
                break
        if dataset[dataset[split_attribute] == split_value]['Wait'].iloc[0] == 'T':
            O = 'T'
        else:
            O = 'F'
        dataset = dataset.drop(dataset[dataset[split_attribute] == split_value].index)
        return (split_attribute, split_value, O), decision_list(dataset)
'''
