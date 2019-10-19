import numpy as np

training_data = np.loadtxt('optdigits_train.txt', delimiter=',')
testing_data = np.loadtxt('optdigits_test.txt', delimiter=',')
'''
def myLDA(data, num_principal_components):
    data_dict, in_class_scatter = {}, {}
    total_mean = np.zeros((64,64))
    total_within_class = np.zeros((64,64))
    between_class = np.zeros((64,64))
    for i in range(len(np.unique(data[:,-1]))):
        data_dict[i] = data[data[:,-1]==i][:,:-1]
        in_scatter = np.zeros((64,64))
        class_mean = np.mean(data_dict[i], axis = 0)
        total_mean+=class_mean
        for instance in data_dict[i]:
            product = np.matmul((instance-class_mean), (instance-class_mean).T)
            in_scatter+=product
        total_within_class+=in_scatter
    total_mean/=10
    
    for i in range(10):
        between_class+=len(data_dict[i])*np.matmul((np.mean(data_dict[i], axis = 0) - total_mean), (np.mean(data_dict[i], axis = 0) - total_mean).T)
    
    W = np.matmul(np.linalg.pinv(total_within_class), between_class)
    e_values, e_vectors = np.linalg.eigh(W)
    e_values = np.flip(e_values)
    e_vectors = np.flip(e_vectors, axis = 1)
    W = e_vectors[:,:num_principal_components]
    
    return e_values, e_vectors
'''

def myLDA(data, num_principal_components):
    data_list, in_class_scatter = [], np.zeros((64,64))
    total_mean = np.zeros((64,64))
    total_within_class = np.zeros((64,64))
    between_class = np.zeros((64,64))
    class_means = []
    # calculate the 
    for i in range(10):
        data_list.append(data[data[:,-1]==1][:,:-1])
        class_mean = np.mean(data_list[i], axis = 0)
        total_mean += class_mean
        class_means.append(class_mean)
        within_class = np.zeros((64,64))
        for j in range(len(data_list[i])):
            within_class+=np.matmul((data_list[i][j]-class_mean), (data_list[i][j]-class_mean).T)
        total_within_class+=within_class
    total_mean = total_mean/10
    for i in range(10):
        between_class+=len(data_list[i])*np.matmul((class_means[i]-total_mean), (class_mean[i]-total_mean).T)
    e_values, e_vectors = np.linalg.eigh(np.matmul(np.linalg.pinv(total_within_class), between_class))
    e_values = np.flip(e_values)
    e_vectors = np.flip(e_vectors, axis = 1)

    return e_vectors[:,:num_principal_components]
