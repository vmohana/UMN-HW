'''
INTRODUCTION TO MACHINE LEARNING - HW2
NAME: MOHANA KRISHNA VUTUKURU
MULTIVARIATE GAUSSIAN DISTRIBUTION LEARNING
'''
# Importing the libraries
import numpy as np

training_data = np.loadtxt('training_data1.txt', delimiter=',')
test_data = np.loadtxt('test_data1.txt', delimiter=',')

def discriminant_1(S, instance, mean, prior, model):
    if model==1:
        term1 = -0.5*np.log(np.linalg.det(S)) 
        term2 = - 0.5*np.matmul((instance-mean).T, np.matmul(np.linalg.pinv(S), (instance-mean))) + np.log(prior)
        return  term1+term2
    if model==2:
        term1 = -0.5*np.matmul((instance-mean).T, np.matmul(np.linalg.pinv(S), (instance-mean))) + np.log(prior)
        return term1

    if model == 3:
        term1 = -0.5*np.log(np.linalg.det(S)) 
        term2 = - 0.5*np.matmul((instance-mean).T, np.matmul(np.linalg.pinv(S), (instance-mean))) + np.log(prior)
        return  term1+term2     




def MultiGaussian(training_data, testing_data, model):
    
    class_data_1 = training_data[training_data[:,-1]==1]
    class_data_2 = training_data[training_data[:,-1]==2]

    X_train_1 = class_data_1[:,:-1]
    y_train_1 = class_data_1[:,-1]
    X_train_2 = class_data_2[:,:-1]
    y_train_2 = class_data_2[:,-1]
    X_test = test_data[:,:-1]
    y_test = test_data[:,-1]

    pc_1 = len(class_data_1)/len(training_data)
    pc_2 = 1 - pc_1

    mean_1 = np.reshape(np.mean(X_train_1, axis = 0), (8,1))
    mean_2 = np.reshape(np.mean(X_train_2, axis = 0), (8,1))
    

    S1 = np.zeros((8,8))
    S2 = np.zeros((8,8))

    for i in range(len(X_train_1)):
        X = np.reshape(X_train_1[i], (8,1)) - np.reshape(mean_1, (8,1)) 
        S1+= np.matmul(X, X.T)
    #print(S1)
    for i in range(len(X_train_2)):
        X = np.reshape(X_train_2[i], (8,1)) - np.reshape(mean_2, (8,1)) 
        S2+= np.matmul(X, X.T)
    #print(S2)
    S1 /= len(class_data_1)
    S2 /= len(class_data_2)
    
    if model == 1:
        predictions = []
        for instance in X_test:
            g1 = discriminant_1(S1, np.reshape(instance, (8,1)), mean_1, pc_1, 1)
            g2 = discriminant_1(S2, np.reshape(instance, (8,1)), mean_2, pc_2, 1)
            
            if g1>g2:
                predictions.append(1)
            else:
                predictions.append(2)
        error = 0
        for i in range(len(predictions)):
            if predictions[i] != y_test[i]:
                error+=1
        print(error/len(predictions))


    elif model == 2:
        predictions = []
        error = 0
        S = S1*pc_1 + S2*pc_2
        for instance in X_test:
            g1 = discriminant_1(S, np.reshape(instance, (8,1)), mean_1, pc_1, 1)
            g2 = discriminant_1(S, np.reshape(instance, (8,1)), mean_2, pc_2, 1)
            if g1>g2:
                predictions.append(1)
            else:
                predictions.append(2)            
        print(predictions)
        for i in range(len(predictions)):
            if predictions[i] != y_test[i]:
                error+=1
        print(error/len(predictions))

    else:
        alpha1 = 0
        alpha2 = 0
        error = 0
        predictions = []
        for i in range(8):
            for t in range(len(X_train_1)):
                alpha1+=(X_train_1[t][i] - mean_1[i])
        for i in range(8):
            for t in range(len(X_train_2)):
                alpha2+=(X_train_2[t][i] - mean_2[i])
        
        alpha1 /= (len(X_train_1)*8)
        alpha2 /= (len(X_train_2)*8)

        S1 = np.identity(8)*alpha1
        S2 = np.identity(8)*alpha2
        for instance in X_test:
            g1 = discriminant_1(S1, np.reshape(instance, (8,1)), mean_1, pc_1, 1)
            g2 = discriminant_1(S2, np.reshape(instance, (8,1)), mean_2, pc_2, 1)
            if g1>g2:
                predictions.append(1)
            else:
                predictions.append(2)            
        print(predictions)           
        for i in range(len(predictions)):
            if predictions[i] != y_test[i]:
                error+=1
        print(error/len(predictions))
    return S1.shape, S2.shape

print(MultiGaussian(training_data, test_data, 1))