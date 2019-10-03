# Python code for bayesian learning

# Import libraries 
import numpy as np

# Reading and extracting data
training_data = np.loadtxt('training_data.txt')
validation_data = np.loadtxt('validation_data.txt')
#testing_data = np.loadtxt('testing_data.txt')

def Bayes_learning(training_data, validation_data):
    '''
    Choose the best prior
    '''

    error_record = []
    error_sigma = []

    sigma_values = [0.00001,0.0001,0.001,0.01,0.1,1,2,3,4,5,6]
    # First, let's split the dataset into two parts according to the target variable
    c1 = np.array([x for x in training_data if x[-1]==1]) # All the data with class 1
    c2 = np.array([x for x in training_data if x[-1]==2]) # All the data with class 2

    # The response column for each of these parts
    c1_response = c1[:, -1]
    c2_response = c2[:, -1]

    # Select the first 100 columns in each part.
    c1 = c1[:, :100]
    c2 = c2[:, :100]

    # The lists below store the estimator values for each class
    estimates_1 = []
    estimates_2 = []

    # Calculate the estimate for each class
    for i in range(100):
        estimates_1.append(np.sum(c1[:, i]*c1_response)/np.sum(c1_response))
        estimates_2.append(np.sum(c2[:, i]*c2_response)/np.sum(c2_response))

    # Calculate likelihood values and predictions on the validation set
    error_record=[]
    sigmas = []
    for sigma in sigma_values:
        error = 0
        for instance in validation_data:
            pxc1 = 1
            pxc2 = 1
            prediction = 0 
            for j in range(100):
                pxc1 *= (estimates_1[j]**instance[j])*((1-estimates_1[j])**(1-instance[j]))
                pxc2 *= (estimates_2[j]**instance[j])*((1-estimates_2[j])**(1-instance[j]))
            pc1 = pxc1*(1-np.exp(-sigma))
            pc2 = pxc2*(np.exp(-sigma))
            if pc1>pc2:
                prediction = 1
            else:
                prediction = 2
            if prediction!=instance[-1]:
                error+=1
        sigmas.append(sigma)
        error_record.append(error/200)
        
    # Get the sigma value with the best error
    lowest_sigma = [s for _, s in sorted(zip(error_record, sigmas))]
    #print(lowest_sigma)
    print('VALIDATION ERROR RATES\n--------------------------\n')
    for i in range(len(sigmas)):
        print('Sigma value:', sigmas[i], '|\t Error:', error_record[i]*100, '%')
    return estimates_1, estimates_2, (1-np.exp(-lowest_sigma[0])), np.exp(-lowest_sigma[0]) 
            

#print(Bayes_Learning(training_data, validation_data))