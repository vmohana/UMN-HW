# Correct likelyhood weighting

import numpy as np
import sys

num_samples  = int(sys.argv[1])
num_steps = int(sys.argv[2])
evidence = sys.argv[3:13]
evidence = [int(i) for i in evidence]

T = {1: np.array([0.7, 0.3]), 0: np.array([0.3, 0.7])}
E = {1: np.array([0.9, 0.2]), 0: np.array([0.1, 0.8])}

def filtering(time_slice, transition_matrix, evidence_matrix, evidence):
    # Recursive base call
    if time_slice==0:
        return np.array([0.5, 0.5])
    # Recursive call
    else:
        prediction = transition_matrix[1]*filtering(time_slice-1, transition_matrix, evidence_matrix, evidence)[0] + \
             transition_matrix[0]*filtering(time_slice-1, transition_matrix, evidence_matrix, evidence)[1]
        update = evidence_matrix[evidence[time_slice]]*prediction
        return update/np.sum(update)

def lw(T, E, evidence, num_steps, num_samples):
    samples = []
    weights = []
    for i in range(num_samples):
        weight = 1
        sample = np.random.choice([1,0], p = [0.5, 0.5])
        pass_sample = []
        for j in range(num_steps):
            sample = np.random.choice([1,0], p = [T[1][1-sample], T[0][1-sample]])
            pass_sample.append(sample)
            weight *= E[evidence[j]][1-sample]
        samples.append(pass_sample)
        weights.append(weight)
    positive_weights = 0
    for i in range(len(samples)):
        if samples[i][-1] == 1:
            positive_weights+=weights[i]
    return positive_weights/np.sum(weights)

print('Filtered estimate for R10:', filtering(9, T, E, evidence))

print('Mean estimate of P(R10|E) from likelihood weighting after 100 trials:',np.mean([lw(T, E, evidence, num_steps, num_samples) for i in range(100)]))
print('Variance estimate P(R10|E) from likelihood weighting after 100 trials:',np.var([lw(T, E, evidence, num_steps, num_samples) for i in range(100)]))
