import numpy as np
from Bayes_Learning import Bayes_learning

testing_data = np.loadtxt('testing_data.txt')
training_data = np.loadtxt('training_data.txt')
validation_data = np.loadtxt('validation_data.txt')

p1, p2, pc1, pc2 = Bayes_learning(training_data, validation_data)

def Bayesian_testing(testing_data, p1, p2, pc1, pc2):
    error = 0
    for instance in testing_data:
        pcx1 = 1
        pcx2 = 1
        prediction = 0
        for j in range(100):
            pcx1 *= (p1[j]**instance[j])*((1-p1[j])**(1-instance[j]))
            pcx2 *= (p2[j]**instance[j])*((1-p2[j])**(1-instance[j]))

        C1 = pcx1*pc1
        C2 = pcx2*pc2

        if C1>C2:
            prediction=1
        else:
            prediction=2
        if prediction!=instance[-1]:
            error+=1

    print('Test set error with optimal sigma:', (error/2), '%')

Bayesian_testing(testing_data, p1, p2, pc1, pc2)