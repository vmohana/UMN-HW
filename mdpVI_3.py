# Value iteration where terminal states have utilities and the agent can visit terminal states. 
# Terminal states have 0 starting utility.

import sys

reward = float(sys.argv[1])

def get_reward(state, reward):
    if state == (4,2):
        return -1
    elif state == (4,3):
        return 1
    else:
        return reward
def get_new_state(state, action):
    
    if action == 'u':
        if state[1] + 1 > 3 or(state[1] == 1 and state[0] == 2):
            return state
        else:
            return (state[0], state[1]+1)
    elif action == 'd':
        if state[1] - 1 < 1 or(state[1] == 3 and state[0]==2):
            return state
        else:
            return (state[0], state[1]-1)
    elif action == 'l':
        if state[0] - 1 < 1 or (state[0] == 3 and state[1]==2):
            return state
        else:
            return (state[0]-1, state[1])
    else:
        if state[0] + 1 > 4 or (state[0] == 1 and state[1]==2):
            return state
        else:
            return (state[0]+1, state[1])

def insert_probability(state, possibility_dict, probability):
    if state in list(possibility_dict.keys()):
        possibility_dict[state] += probability
    else:
        possibility_dict[state] = probability
    return possibility_dict    

def get_state_probabilities(state, action):
    possible_states = {}
    if action == 'u':
        possible_states = insert_probability(get_new_state(state, action), possible_states, 0.8)
        possible_states = insert_probability(get_new_state(state, 'l'), possible_states, 0.1)
        possible_states = insert_probability(get_new_state(state, 'r'), possible_states, 0.1)
    
    elif action == 'd':
        possible_states = insert_probability(get_new_state(state, action), possible_states, 0.8)
        possible_states = insert_probability(get_new_state(state, 'l'), possible_states, 0.1)
        possible_states = insert_probability(get_new_state(state, 'r'), possible_states, 0.1)
        
    elif action == 'l':
        possible_states = insert_probability(get_new_state(state, action), possible_states, 0.8)
        possible_states = insert_probability(get_new_state(state, 'u'), possible_states, 0.1)
        possible_states = insert_probability(get_new_state(state, 'd'), possible_states, 0.1)
    else:
        possible_states = insert_probability(get_new_state(state, action), possible_states, 0.8)
        possible_states = insert_probability(get_new_state(state, 'u'), possible_states, 0.1)
        possible_states = insert_probability(get_new_state(state, 'd'), possible_states, 0.1)
        
    return possible_states, list(possible_states.keys())
'''
    
def get_state_probabilities(state, action):
    state_probability = None
    if action == 'u':
        state_probability = {get_new_state(state, action):0.8, get_new_state(state, 'l'):0.1, get_new_state(state, 'r'):0.1}
    elif action == 'd':
        state_probability = {get_new_state(state, action):0.8, get_new_state(state, 'l'):0.1, get_new_state(state, 'r'):0.1}
    elif action == 'l':
        state_probability = {get_new_state(state, action):0.8, get_new_state(state, 'u'):0.1, get_new_state(state, 'd'):0.1}
    else:
        state_probability = {get_new_state(state, action):0.8, get_new_state(state, 'u'):0.1, get_new_state(state, 'd'):0.1}
    return state_probability, list(state_probability.keys())
'''

def maximum_utility(utilities, state):
    action_utilities = []
    for action in ['u','d','l','r']:
        probabilities, possibilities = get_state_probabilities(state, action)
        #print(probs, poss)
        #print(action, probabilities)
        action_util = 0
        for p in possibilities:
            #print(p)
            action_util += probabilities[p]*utilities[p]
        action_utilities.append(action_util)
        #print(action_util)
    return max(action_utilities)

def value_iteration(reward):
    utilities = {(1,1):0, (1,2):0, (1,3):0, (2,1):0, (2,3):0, (3,1):0, (3,2):0, (3,3):0, (4,1):0, (4,2):-1, (4,3):1}
    updated_utilities = utilities

    while True:
        utilities = updated_utilities
        delta = 0
        gamma = 0.9
        print(utilities)
        for state in list(utilities.keys()):
            updated_utilities[state] = get_reward(state, reward) + gamma*(maximum_utility(utilities, state))

value_iteration(reward)
