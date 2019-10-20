import numpy as np

training_data = np.loadtxt('optdigits_train.txt', delimiter = ',')


def myLDA(data, principal_components):
    data_dict, scatter = {}, {}
    total_mean = np.mean(data[:,:-1], axis = 0)
    class_means = {}
    between_class_matrix = np.zeros((64,64))
    for i in range(10):
        data_dict[i] = data[data[:,-1]==i][:,:-1]
        class_mean = np.mean(data_dict[i], axis = 0)
        class_means[i] = class_mean
        in_class_scatter = np.zeros((64,64))
        for instance in data_dict[i]:
        	in_class_scatter+=np.matmul((instance - class_mean), (instance - class_mean).T)
        scatter[i]=in_class_scatter

    total_scatter = np.zeros((64,64))
    for i in range(10):
    	total_scatter+=scatter[i]

    for i in range(10):
    	between_class_matrix+=len(data_dict[i])*np.matmul((class_means[i]-total_mean), (class_means[i]-total_mean).T)

    W = np.matmul(np.linalg.pinv(total_scatter), between_class_matrix)
    e_values, e_vectors = np.linalg.eigh(W)
    e_vectors = np.flip(e_vectors, axis = 1)
    return e_values, e_vectors[:,:principal_components]


print(myLDA(training_data, 5))
