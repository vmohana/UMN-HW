'''
ARTIFICIAL INTELLIGENCE - HOMEWORK 2
NAME: MOHANA KRISHNA VUTUKURU
'''
import sys
import numpy as np

number_of_evidence = int(sys.argv[1])
evidence = sys.argv[2:12]
evidence = [int(i) for i in evidence]

#0->True
#1-> False
T = {0: np.array([0.7, 0.3]), 1: np.array([0.3, 0.7])}
#1->True
#0->False
E = {1: np.array([0.9, 0.2]), 0: np.array([0.1, 0.8])}

def normalize(p_vector):
    return p_vector/(p_vector[0]+p_vector[1])
    

def filtering(time_slice, transition_matrix, evidence_matrix, evidence):

    if time_slice>len(evidence):
        return 'ERROR:- time_slice > len(evidence)'

    # Recursive base call
    if time_slice==0:
        return np.array([0.5, 0.5])

    # Recursive call
    else:
        prediction = transition_matrix[0]*filtering(time_slice-1, transition_matrix, evidence_matrix, evidence)[0] + \
             transition_matrix[1]*filtering(time_slice-1, transition_matrix, evidence_matrix, evidence)[1]
        update = evidence_matrix[evidence[time_slice]]*prediction
        return normalize(update)
        

def smoothing(k, transition_matrix, evidence_matrix, evidence):
    # Base case
    if k==len(evidence):
        return 1
    # Recursive call
    else:
        recursive_value = smoothing(k+1, transition_matrix, evidence_matrix, evidence)
        return np.array(evidence_matrix[evidence[k]][0]*recursive_value*transition_matrix[0]) + \
            np.array(evidence_matrix[evidence[k]][1]*recursive_value*transition_matrix[1])
    

#print(filtering(1, T, E, evidence))
#print(smoothing(1, T, E, [1,1]))
#print(normalize(filtering(1,T,E,[1,1])*smoothing(1, T, E, [1,1])))
for i in range(number_of_evidence):
    print('Smoothed estimates for X'+str(i)+':',normalize(filtering(i, T, E, evidence)*smoothing(i, T, E, evidence)))
