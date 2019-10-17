'''
NAME: MOHANA KRISHNA VUTUKURU
ARTIFICIAL INTELLIGENCE 2
LIKELIHOOD WEIGHTING
'''

import numpy as np 
import sys

num_samples = int(sys.argv[1])
num_steps = int(sys.argv[2])
evidence = sys.argv[3:13]
evidence = [int(i) for i in evidence]

T = {1: np.array([0.7, 0.3]), 0: np.array([0.3, 0.7])}
E = {1: np.array([0.9, 0.2]), 0: np.array([0.1, 0.8])}

# Code for filtering from SmoothingHMM.py 
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

filtered_estimate_10 = filtering(num_steps-1, T, E, evidence)
print(filtered_estimate_10)

# The Bayesian net representation can be obtained by ignoring the time component. R9->R10->E10
# Topology is [R9, R10, E10]. R9, R10 -> non evidence variables, have to be sampled. Prior for R9 is obtained 
# through filtering
prior_R9 = filtering(8, T, E, evidence)
'''
W = {1:0, 0:0}
for i in range(num_samples):
    r9_sample, r10_sample = None, None
    weight = 1
    for j in range(3):
        if j == 0:
            r9_sample = np.random.choice([1,0], p = prior_R9)
            #print(r9_sample)
        if j == 1:
            r10_sample = np.random.choice([1,0], p = [T[1][1-r9_sample], T[0][1-r9_sample]])
            #print(r10_sample)
        if j == 2:
            weight*=E[evidence[-1]][1-r10_sample]
    W[r10_sample]+=weight
'''
def lw(num_samples, T, E):
    W = {1:0, 0:0}
    for i in range(num_samples):
        r9_sample, r10_sample = None, None
        weight = 1
        for j in range(3):
            if j == 0:
                r9_sample = np.random.choice([1,0], p = prior_R9)
                #print(r9_sample)
            if j == 1:
                r10_sample = np.random.choice([1,0], p = [T[1][1-r9_sample], T[0][1-r9_sample]])
                #print(r10_sample)
            if j == 2:
                weight*=E[evidence[-1]][1-r10_sample]
        W[r10_sample]+=weight    
    probabilities = np.array(list(W.values()))
    return [probabilities[0]/np.sum(probabilities), probabilities[1]/np.sum(probabilities)]

print(lw(num_samples, T, E))