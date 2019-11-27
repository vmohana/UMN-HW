'''
INTRODUCTION TO MACHINE LEARNING
ASSIGNMENT - 4
AUTHOR: MOHANA KRISHNA
'''
import numpy as np
import matplotlib.pyplot as plt

def softmax(prediction):
    exp_prediction = np.exp(prediction)
    return exp_prediction/sum(exp_prediction)

def one_hot(responses, K):
    encoded_responses = []
    for response in responses:
        encoded_response = []
        for i in range(K):
            if i == response:
                encoded_response.append(1)
            else:
                encoded_response.append(0)
        encoded_responses.append(encoded_response)
    return np.array(encoded_responses)

def ReLU(activations):
    for activation in range(len(activations)):
        if activations[activation]  < 0:
            activations[activation] = 0
    return activations
    
def MLPtest(test_data, W, V):
    test_data = np.loadtxt('optdigits_test.txt', delimiter=',')
    x_test = test_data[:,:-1]
    y_test = one_hot(test_data[:,-1],10)
    W = W.T
    W_bias = W[:,-1]
    W = W[:,:-1]
    V = V.T
    V_bias = V[:,-1]
    V = V[:,:-1]

    error = 0
    Z_matrix = []
    for t in range(len(x_test)):
        instance = np.reshape(x_test[t], (-1,1))
        response = np.reshape(y_test[t], (-1,1))

        Z = np.matmul(np.hstack((np.reshape(W_bias, (len(W_bias), 1)), W)), np.vstack(([1],instance)))
        Z_relu = ReLU(Z)
        Z_matrix.append(np.reshape(Z_relu, (1,18)))

        prediction = np.matmul(np.hstack((np.reshape(V_bias, (len(V_bias), 1)), V)), np.vstack(([1],Z_relu)))
        prediction = softmax(prediction)

        if np.argmax(prediction) != np.argmax(response):
            error += 1
    print('Test error:',error/len(x_test))
    return Z_matrix
