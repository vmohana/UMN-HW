'''
INTRODUCTION TO MACHINE LEARNING
ASSIGNMENT - 4
AUTHOR: MOHANA KRISHNA
'''

import numpy as np
import matplotlib.pyplot as plt

def relu_backward(dl_dy, x, y):
    # TO DO
    dl_dx = dl_dy
    dl_dx[x < 0] = 0
    return dl_dx

def r_backward(error, activation):
    error[activation < 0] = 0
    return error

def backward(error, weight, bias, instance, prediction):
    
    dinstance = []
    for w in range(len(weight)):
        dinstance.append(error[w][0]*weight[w])
    dinstance = np.array(dinstance)
    dinstance = np.reshape(np.sum(dinstance, axis = 0), (-1,1))
    dweight = np.matmul(error, np.transpose(instance))
    return dweight, dinstance, error


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

def MLPtrain(train_data, val_data, K, H):
    
    # Read the data and one-hot encode the responses 
    train_data = np.loadtxt(train_data, delimiter=',')
    val_data = np.loadtxt(val_data, delimiter=',')
    x_train, y_train = train_data[:,:-1], one_hot(train_data[:,-1], K)
    x_val, y_val = val_data[:,:-1], one_hot(val_data[:,-1], K)
    
    # Initialize weight matrices and bias weights
    W = np.random.randn(H, x_train.shape[1])*0.01
    W_bias = np.random.randn(H,1)
    V = np.random.randn(K, H)*0.01
    V_bias = np.random.randn(K,1)

    epoch_loss = []
    epoch_accuracy = []
    learning_rate = 0.001

    # Train the MLP for 200 epochs
    for epoch in range(300):
        #print('Epoch number:',epoch)
        if epoch%20 == 0:
            learning_rate *= 0.1

        # Initialize the delta values
        delta_W = 0
        delta_W_bias = 0
        delta_V = 0
        delta_V_bias = 0
        losses = []
        accuracy = 0

        
        for t in range(len(train_data)):
            
            # Reshape instance and response and add the bias term
            instance = np.reshape(x_train[t], (-1,1))
            response = np.reshape(y_train[t], (-1,1))
            #print('response',np.argmax(response))
            
            # Calculate hidden layer values
            Z = np.matmul(np.hstack((W_bias, W)), np.vstack(([1],instance)))
            Z_relu = ReLU(Z)
            

            # Calculate the final layer activations and apply softmax 
            prediction = np.matmul(np.hstack((V_bias, V)), np.vstack(([1],Z_relu)))
            prediction = softmax(prediction)

            # Calculate error and loss
            loss = -1*np.sum(response*np.log(prediction)) 
            error = prediction - response
            losses.append(loss)      

            if np.argmax(response) == np.argmax(prediction):
                accuracy += 1

            # Backpropagation

            # Backpropagate through the output layer
            d_V, d_Z, d_V_B = backward(error, V, V_bias, Z_relu, prediction)
            delta_V += d_V
            delta_V_bias += d_V_B

            # Backpropagate through the ReLU and the hidden layer
            d_Relu_Z = r_backward(d_Z, Z)
            d_W, d_X, d_W_B = backward(d_Relu_Z, W, W_bias, instance, Z)
            delta_W += d_W
            delta_W_bias += d_W_B
        
        epoch_accuracy.append(accuracy/len(x_train))
        
        # Update the weight matrix
        W = W - learning_rate*delta_W
        W_bias = W_bias - learning_rate*delta_W_bias
        V = V - learning_rate*delta_V
        V_bias = V_bias - learning_rate*delta_V_bias
        epoch_loss.append(np.mean(losses))
    
    print('Training error for h={} after {} epochs:'.format(H, len(epoch_loss)), np.mean(epoch_loss))

    # Select the model using the validation set on the learned weights


            
    return epoch_accuracy

for h in [12,15,18]:
    MLPtrain('optdigits_train.txt', 'optdigits_valid.txt', 10, h)