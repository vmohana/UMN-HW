import pandas as pd

# Preprocess the dataset
dataset = pd.read_csv('dataset.csv', header=None)
dataset.columns = ['Alt', 'Bar', 'Fri', 'Hun', 'Pat', 'Price', 'Rain', 'Res', 'Type', 'Est', 'Wait']
dataset[dataset.columns] = dataset.apply(lambda x: x.str.strip())

def split_data(dataset, column):
    split_data = {}
    for value in dataset[column].unique():
        split_data[value] = dataset[dataset[column] == value]
    return split_data

def pure(dataset):
    if len(dataset['Wait'].unique()) == 1:
        return True
    else:
        return False


def decision_list(dataset):
    if len(dataset) == 0:
        return 
    else:
        split_attribute = None
        split_value = None
        for attribute in dataset.columns[0:10]:
            sub_data = split_data(dataset, attribute)
            for value in list(sub_data.keys()):
                split = None
                if pure(sub_data[value]):
                    split_attribute = attribute
                    split_value = value
                    break
                break
        if dataset[dataset[split_attribute] == split_value]['Wait'].iloc[0] == 'T':
            O = 'Yes'
        else:
            O = 'No'
        print(split_value, split_attribute, O)
        
        dataset = dataset.drop(dataset[dataset[split_attribute] == split_value].index)
        print(dataset)
        return (split_attribute, split_value, O), decision_list(dataset)

dlist = decision_list(dataset)
print(dlist)
print(dlist[0], dlist[1][0])
