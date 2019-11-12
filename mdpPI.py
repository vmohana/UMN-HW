# This script follows the pseudocode highlighted in Rich Sutton's book
import sys

reward = float(sys.argv[-1])

def get_reward(state, reward):
    if state == (4,2):
        return -1
    elif state == (4,3):
        return 1
    else:
        return reward

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

def maximum_utility(utilities, state, gamma, reward):
    action_utilities = []
    for action in ['u','d','l','r']:
        probabilities, possibilities = get_state_probabilities(state, action)
        action_util = 0
        #print(probabilities)
        for p in possibilities:
            
            action_util += probabilities[p]*(get_reward(p,reward) + gamma*utilities[p])
            #print(action, p, action_util)
        action_utilities.append(action_util)
        #print(action_util)
    return max(action_utilities)

print(get_state_probabilities((3,3), 'u'))

def policy_evaluation(states, policy, utilities, reward, gamma):
    for state in states:
        state_probabilities, possibilities = get_state_probabilities(state, policy[state])

        expected_utility = 0
        for possible_state in possibilities:
            expected_utility += state_probabilities[possible_state]*utilities[possible_state]
        utilities[state] = get_reward(state, reward) + gamma*expected_utility
        print(utilities)
    return utilities
'''

def policy_iteration(reward):

    # Initialize utilities and policy
    utilities = {(1,1):0, (1,2):0, (1,3):0, (2,1):0, (2,3):0, (3,1):0, (3,2):0, (3,3):0, (4,1):0, (4,2):0, (4,3):0}
    terminal_states = [(4,2), (4,3)]
    non_terminal_states = [item for item in list(utilities.keys()) if item not in terminal_states]
    policy = {(1,1):'d', (1,2):'l', (1,3):'u', (2,1):'u', (2,3):'u', (3,1):'l', (3,2):'d', (3,3):'l', (4,1):'u'}

    return non_terminal_states
'''

def policy_iteration(reward):
    utilities = {(1,1):0, (1,2):0, (1,3):0, (2,1):0, (2,3):0, (3,1):0, (3,2):0, (3,3):0, (4,1):0, (4,2):0, (4,3):0}
    terminal_states = [(4,2), (4,3)]
    non_terminal_states = [item for item in list(utilities.keys()) if item not in terminal_states]
    gamma = 0.95
    policy = {(1,1):'d', (1,2):'l', (1,3):'u', (2,1):'u', (2,3):'u', (3,1):'l', (3,2):'d', (3,3):'l', (4,1):'u'}
    while True:
        utilities = policy_evaluation(non_terminal_states, policy, utilities, reward, gamma)
        unchanged = True
        for state in non_terminal_states:
            
            # Calculate possible state policy utility 
            policy_utility = 0
            state_probabilities, possibilities = get_state_probabilities(state, policy[state]) 
            for possible_states in possibilities:
                policy_utility += state[possible_states]*utilities[possible_states]
            
            


'''    
utilities = {(1,1):0, (1,2):0, (1,3):0, (2,1):0, (2,3):0, (3,1):0, (3,2):0, (3,3):0, (4,1):0, (4,2):0, (4,3):0}
terminal_states = [(4,2), (4,3)]
policy = {(1,1):'d', (1,2):'l', (1,3):'u', (2,1):'u', (2,3):'u', (3,1):'l', (3,2):'d', (3,3):'l', (4,1):'u'}
non_terminal_states = [item for item in list(utilities.keys()) if item not in terminal_states]
print(policy_evaluation(non_terminal_states, policy, utilities, reward, 0.95))
'''
