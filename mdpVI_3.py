# This script follows the pseudocode described in Rich Sutton's reinforcement learning book
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

#print(maximum_utility(utilities, (3,3), 1, reward))


def calculate_policy(state, utilities, gamma, reward):
    action_utilities = {}
    for action in ['u','d','r','l']:
        probabilities, possibilities = get_state_probabilities(state, action)
        action_utility = 0
        for possible_state in possibilities:
            #probabilities[possible_state]
            action_utility += probabilities[possible_state]*(get_reward(possible_state, reward) + gamma*utilities[possible_state])
        action_utilities[action] = action_utility
    sorted_utilities = sorted(action_utilities.items(), key = lambda kv: kv[1])
    return sorted_utilities[-1][0]


def value_iteration(reward):
    utilities = {(1,1):0, (1,2):0, (1,3):0, (2,1):0, (2,3):0, (3,1):0, (3,2):0, (3,3):0, (4,1):0, (4,2):0, (4,3):0}
    terminal_states = [(4,2),(4,3)]
    gamma = 0.95
    non_terminal_states = [item for item in list(utilities.keys()) if item not in terminal_states]
    print(utilities)
    epsilon = 0.00001

    while True:
        delta = 0
        print(utilities)
        for state in list(non_terminal_states):
            value = utilities[state]
            utilities[state] = maximum_utility(utilities, state, gamma, reward)
            delta = max(abs(value - utilities[state]), delta)
        if delta < epsilon*(1-gamma)/gamma:
            break

    print('POLICIES FOR GAMMA={} | EPSILON={}'.format(gamma, epsilon))
    for state in non_terminal_states:
        print('policy for {}: {}'.format(state, calculate_policy(state, utilities, gamma, reward)))
    return utilities

utilities = value_iteration(reward)

print(calculate_policy((3,3), utilities, 0.99, reward))
