# Transition working well

import sys

reward = float(sys.argv[1])

# Function to get the new state given the current state and action taken
def get_new_state(state, action):
    new_state = None
    if action == 'u':
        if state[1] + 1 > 3 or(state[1]+1 == 2 and state[0] == 2) or (state[1]+1 == 3 and state[0] == 4) or (state[1]+1 == 2 and state[0] == 4):
            return state
        else:
            return (state[0], state[1]+1)
    elif action == 'd':
        if state[1] - 1 < 1 or(state[1]-1 == 2 and state[0]==2):
            return state
        else:
            return (state[0], state[1]-1)
    elif action == 'l':
        if state[0] - 1 < 1 or (state[0] - 1 == 2 and state[1]==2):
            return state
        else:
            return (state[0]-1, state[1])
    else:
        if state[0] + 1 > 4 or (state[0] + 1 == 2 and state[1]==2) or (state[0] + 1 == 4 and state[1]==3) or (state[0] + 1 == 4 and state[1]==2):
            return state
        else:
            return (state[0]+1, state[1])
    return new_state

def get_expected_utility(state, utilities):
    expected_utilites = []
    for action in ['u','d','l','r']:
        if action == 'u':
            other_actions = ['l','r']
            expected_utility = 0.8*utilities[get_new_state(state, action)]\
                + 0.1*utilities[get_new_state(state, other_actions[0])]\
                    + 0.1*utilities[get_new_state(state, other_actions[1])]
            expected_utilites.append(expected_utility) 
        elif action == 'd':
            other_actions = ['l','r']
            expected_utility = 0.8*utilities[get_new_state(state, action)]\
                +0.1*utilities[get_new_state(state, other_actions[0])]\
                    + 0.1*utilities[get_new_state(state, other_actions[1])]
            expected_utilites.append(expected_utility)
        elif action == 'l':
            other_actions = ['d', 'u']
            expected_utility = 0.8*utilities[get_new_state(state, action)]\
                +0.1*utilities[get_new_state(state, other_actions[0])]\
                    +0.1*utilities[get_new_state(state, other_actions[1])]
            expected_utilites.append(expected_utility)
        else:
            other_actions = ['d','u']
            expected_utility = 0.8*utilities[get_new_state(state, action)]\
                +0.1*utilities[get_new_state(state, other_actions[0])]\
                    +0.1*utilities[get_new_state(state, other_actions[1])]
            expected_utilites.append(expected_utility)
    return max(expected_utilites)

def get_maximum_utility(state, utilities, reward, gamma):
    action_utilities = []
    for action in ['u','d','l','r']:
        state_possibilites = get_state_probability(state, action)
        #print(state_possibilites)
        action_utility = 0
        for possible_state in list(state_possibilites.keys()):
            action_utility += state_possibilites[possible_state]*utilities[possible_state]     
        action_utilities.append(action_utility)     
    return max(action_utilities)

def insert_probability(state, possibility_dict, probability):
    if state in list(possibility_dict.keys()):
        possibility_dict[state] += probability
    else:
        possibility_dict[state] = probability
    return possibility_dict    

def get_state_probability(state, action):
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
        
    return possible_states

def value_iteration(reward):
    utilities = {(1,1):0, (1,2):0, (1,3):0, (2,1):0, (2,3):0, (3,1):0, (3,2):0, (3,3):0, (4,1):0, (4,2):-1, (4,3): 1}
    updated_utilities = {(1,1):0, (1,2):0, (1,3):0, (2,1):0, (2,3):0, (3,1):0, (3,2):0, (3,3):0, (4,1):0, (4,2):-1, (4,3): 1}
    terminal_states = [(4,2), (4,3)]
    non_terminal_states = [state for state in list(utilities.keys()) if state not in terminal_states]
    epsilon = 0.1
    gamma = 0.99
    while True:
        utilities = updated_utilities
        delta = 0
        for state in non_terminal_states:
            updated_utilities[state] = reward + gamma*get_maximum_utility(state, utilities, reward, gamma)

            delta = max(delta, abs(updated_utilities[state] - utilities[state]))
        if delta < epsilon*(1-gamma)/gamma:
            return utilities
    
print(value_iteration(reward))