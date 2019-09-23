# Gibbs sampler
'''
ARTIFICIAL INTELLIGENCE - ASSIGNMENT 1: Programming component
NAME: MOHANA KRISHNA VUTUKURU
'''

import numpy as np
import random
import argparse

'''
Algorithm:

1. Start with an arbitrary state.
2. Move to the next state by randomly sampling a value for one of the non-evidence variable X. Sampling of X is done conditioned on the *current values* of the variables in the Markov blanket. The algorithm keeps moving around the search space keeping the evidence variables fixed. 
3. Check how many times each value of the random variable is visited. 
'''

# Getting the number N through user input
parser = argparse.ArgumentParser(description='Number of sampling iterations')
parser.add_argument('integers', metavar = 'N', type = int)
args = parser.parse_args()
N = parser.parse_args().integers

def gibbs_sampler(probabilities, N):

	'''
	Arguments
	----------------
	N: int
		The number of iterations for which we have to run the sampler.

	Returns:
	----------------
	normalized_probability: float
		The probability value P(r=true|s,w)
	'''

	probabilities = {'C':{'R':0.080, '~R':0.020}, 'R':{'C':0.632, '~C':0.180}}
	initial_state = ['C','S','R','W']
	non_evidence_variables = ['C', 'R']
	states = [] # A list that stores the states visited. Each element is in the format [c,s,r,w]
	states.append(initial_state)
	current_state = initial_state

	for i in range(N):

		# Pick a non evidence variable
		ne_variable = random.choice(non_evidence_variables)
		list_index, other_ne_variable = None
		
		if ne_variable=='C':
			other_ne_variable = 'R'
			list_index = 0
		else:
			other_ne_variable = 'C'
			list_index = 2

	#sample_probability = probabilities[ne_variable]


	return normalized_probabilities 
