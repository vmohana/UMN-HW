'''
NAME: V. MOHANA KRISHNA
ARTIFICIAL INTELLIGENCE II - Homework
Coding assignment 

GIBB'S SAMPLER ALGORITHM
-----------------------------------------
1. Start with a random state.
2. Pick a non evidence variable to be sampled ( the non evidence variables must be picked in turn).
3. Sample a value for the picked non evidence variable and change the state as required. 
4. Record how many times the sampler takes each value of the desired non evidence random variable. 
5. Continue for the required number of iterations.
6. Calculate and print the probabilities.  
'''

# Import requried libraries 
import random
import sys
import numpy as np 

# Read argument from command line
N = sys.argv[1]

ne_variables = ['C', 'R'] # The non evidence variables.
probabilities = {'C':{'R':0.4444, '~R':0.04761}, 'R':{'C':0.8148, '~C':0.2157}, '~C':{'R':0.5555, '~R':0.9524}, '~R':{'C':0.1851, '~C':0.7843}} # The probabilities of non evidence variables given their Markov blankets. 
initial_state = ['C','S','R','W'] # An array holding the intial array. It also reflects the current state of random variables. 
false_counter = 0 # The number of times a state in which Rain = false is visited.
true_counter = 0 # The number of times a state with Rain = true is visited.

# For loop for the number of iterations
for i in range(int(N)):
	
	variable_to_sample = ne_variables[i%2] # A variable which records which random variable to sample. 	
	current_state_variable = None # This holds the current value of the random variable.

	# Check the current value of the sampled random variable and store it in current_state_variable. 
	# other_variable records the other random variable which is not being sampled. 
	if variable_to_sample=='C':
		current_state_variable = initial_state[0]
		other_variable = initial_state[2]
	else:
		current_state_variable = initial_state[2]
		other_variable = initial_state[0]

	# Retreive the conditional probability from the dictionary of probabilities. 
	sampling_probability = probabilities[current_state_variable][other_variable]

	# Sample a value according to the probability.
	sampled_value = np.random.choice([True, False], p = [sampling_probability, 1-sampling_probability])

	# Change the state of the sampled random variable if the sampled value is False. Otherwise, keep it same.
	if sampled_value == False:
		if current_state_variable=='C':
			initial_state[0]='~C'
		if current_state_variable=='~C':
			initial_state[0]='C'
		if current_state_variable=='R':
			initial_state[2]='~R'
			false_counter+=1 # Record the number of times a state with Rain = False is visited.
		if current_state_variable=='~R':
			initial_state[2]='R'
			true_counter+=1 # Record the number of times a state with Rain = true is visited.

	# Don't change the value if the sampled value is True. 
	else:
		if current_state_variable=='R':
			true_counter+=1 # Just increment the counter and don't make any changes to the state
		if current_state_variable=='~R':
			false_counter+=1 # Just increment the counter and don't make any changes to the state

print('True: ', true_counter)
print('False', false_counter)
print('probability of Rain = True: ', (true_counter/int(N)))
