'''
NAME: MOHANA KRISHNA VUTUKURU
ARTIFICIAL INTELLIGENEC 2
PARTICLE FILTERING ALGORITHM
'''

import numpy as np
import sys
import random

# Take inputs
num_samples = int(sys.argv[1])
num_steps = int(sys.argv[2])
evidence = sys.argv[3:13]
evidence = [int(i) for i in evidence]

T = {1: np.array([0.7, 0.3]), 0: np.array([0.3, 0.7])}
E = {1: np.array([0.9, 0.2]), 0: np.array([0.1, 0.8])}

def filtering(time_slice, transition_matrix, evidence_matrix, evidence):
    if time_slice>len(evidence):
        return 'ERROR:- time_slice > len(evidence)'
    # Recursive base call
    if time_slice==0:
        return np.array([0.5, 0.5])
    # Recursive call
    else:
        prediction = transition_matrix[1]*filtering(time_slice-1, transition_matrix, evidence_matrix, evidence)[0] + \
             transition_matrix[0]*filtering(time_slice-1, transition_matrix, evidence_matrix, evidence)[1]
        update = evidence_matrix[evidence[time_slice]]*prediction
        return update/np.sum(update)

filtered_estimate_10 = filtering(9, T, E, evidence)
print('Filtered estimate for P(R10|U1..10)=', filtered_estimate_10)
sampling = [0 if i%2==0 else 1 for i in range(num_samples)]

# first sample from [0.5, 0.5]
#print(sampling)
def particleFiltering(num_steps, num_samples, T, E):
    sampling = [0 if i%2==0 else 1 for i in range(num_samples)]

    for i in range(num_steps):
        w = []
        e = evidence[i]

        for j in range(num_samples):
            sample = np.random.choice([1,0],p= [T[1][1-sampling[j]], T[0][1-sampling[j]]])
        #print(sample)
            sampling[j] = sample
            w.append(E[e][1-sampling[j]])
    
    #print('W',w,len(w))

        sampling = random.choices(sampling, weights=w, k = num_samples)
    return sampling.count(1)/num_samples
'''
for i in range(num_steps):
    w = []
    e = evidence[i]

    for j in range(num_samples):
        sample = np.random.choice([1,0],p= [T[1][1-sampling[j]], T[0][1-sampling[j]]])
        #print(sample)
        sampling[j] = sample
        w.append(E[e][1-sampling[j]])
    #print('W',w,len(w))

    sampling = random.choices(sampling, weights=w, k = num_samples)
'''
    
#print(sampling)
outputs = []
for i in range(1000):
    outputs.append(particleFiltering(num_steps, num_samples, T, E))
print('After 500 trials of particle filtering with numSamples = {} and numSteps={}:'.format(num_samples, num_steps))
print('Variance of P(R10 | U1..10)=',[np.var(outputs), np.var(1-np.array(outputs))])
print('Mean of P(R10 | U1..10)=',[np.mean(outputs), 1 - np.mean(outputs)])