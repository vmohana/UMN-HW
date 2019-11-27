'''
INTRODUCTION TO MACHINE LEARNING
ASSIGNMENT - 4
AUTHOR: MOHANA KRISHNA
'''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D 


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

Z_matrix = []

train_data = np.loadtxt('optdigits_train.txt', delimiter=',')
val_data = np.loadtxt('optdigits_valid.txt', delimiter=',')
total_data = np.vstack((train_data, val_data))
X, label = total_data[:,:-1], total_data[:,-1]
W = np.load('l1_weight.npy')
W = W.T
W_bias = W[:,-1]
W = W[:,:-1]

for t in range(len(total_data)):
        instance = np.reshape(X[t], (-1,1))
        Z = np.matmul(np.hstack((np.reshape(W_bias, (len(W_bias), 1)), W)), np.vstack(([1],instance)))
        Z_relu = ReLU(Z)
        Z_matrix.append(np.reshape(Z_relu, (1,18)))

Z_matrix = np.reshape(np.array(Z_matrix), (len(total_data), 18))
pca_2 = PCA(n_components=2)
d_2 = pca_2.fit_transform(Z_matrix)

fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(111)
ax.set_title("Hidden layer values",fontsize=14)
ax.set_xlabel("Principal component 1",fontsize=12)
ax.set_ylabel("Principal component 2",fontsize=12)
for i in range(len(total_data)//40):
    ax.text(d_2[i][0], d_2[i][1], str(label[i]), fontsize = 14)
    #ax.annotate(str(label[i]), (d_2[i][0], d_2[i][1]))
ax.grid(True,linestyle='-',color='0.75')
x = d_2[:,0]
y = d_2[:,1]
z = label
# scatter with colormap mapping to z value
ax.scatter(x,y,s=20,c=z, marker = 'o', cmap = cm.jet );
plt.show()

pca_3 = PCA(n_components=3)
d_3 = pca_3.fit_transform(Z_matrix)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(d_3[:,0], d_3[:,1], d_3[:,2],c=z,marker='o',cmap=cm.jet)
ax.set_title("Hidden layer values",fontsize=14)
ax.set_xlabel("Principal component 1",fontsize=12)
ax.set_ylabel("Principal component 2",fontsize=12)
ax.set_zlabel('Principal component 3', fontsize=12)
for i in range(len(total_data)//40):
    ax.text(d_2[i][0], d_2[i][1], d_3[i][2], str(label[i]), fontsize = 14)
plt.show()