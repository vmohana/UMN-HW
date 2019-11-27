from MLPtrain import MLPtrain
from MLPtest import MLPtest
import matplotlib.pyplot as plt
import numpy as np

# Get train and validation errors
training_error = []
validation_error = []
weight_l1 = None
weight_l2 = None
max_error = 0
H = None

for h in [3,6,9,12,15,18]:
    Z, W, V, t_error, v_error = MLPtrain('optdigits_train.txt', 'optdigits_valid.txt', 10, h)
    training_error.append(t_error)
    validation_error.append(v_error)
    if v_error > max_error:
        H = h
        weight_l1 = W
        weight_l2 = V

plt.plot([3,6,9,12,15,18], training_error)
plt.plot([3,6,9,12,15,18], validation_error)
plt.legend(['Training error', 'Validation error'], loc = 'upper left')
plt.title('Error rates')
plt.ylabel('Error')
plt.xlabel('H')
plt.show()

print('The best model has h = {} hidden units'.format(H))
MLPtest('optdigits_test.txt', weight_l1, weight_l2)

np.save('l1_weight.npy', weight_l1)
np.save('l2_weight.npy', weight_l2)