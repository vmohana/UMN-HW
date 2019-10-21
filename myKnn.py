'''
INTRO TO MACHINE LEARNING - HOMEWORK 2
NAME: MOHANA KRISHNA VUTUKURU
'''

import numpy as np
from scipy import spatial

def myKNN(training_data, test_data, k):
    X_train = training_data[:,:-1]
    y_train = training_data[:,-1]
    X_test = test_data[:,:-1]
    y_test = test_data[:,-1]
    predictions = []
    error = 0

    for i in range(len(X_test)):
        distances = spatial.distance.cdist(X_train, [X_test[i]])
        classes = y_train
        closest = [classes for _, classes in sorted(zip(distances, classes))][0:k]
        predictions.append(np.argmax(np.bincount(closest)))

    for p in range(len(predictions)):
        if predictions[p] != y_test[p]:
            error+=1
        else:
            continue
    return (error/len(y_test))*100