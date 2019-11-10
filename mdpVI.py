# Code for value iteration in MDPs
# TO DO 
# Model the state


def get_reward(state, reward):
    if state == (4,2):
        return -1
    elif state == (4,3):
        return 1
    else:
        return reward

def transition_model(state, action):
    # Returns the next state
    new_state = None
    if action == 'u':
        if state[1] + 1 > 3:
            new_state = state
        else:
            new_state = (state[0], state[1]+1)
    elif action == 'd':
        if state[1] - 1 < 1:
            new_state = state
        else:
            new_state = (state[0], state[1]-1)
    elif action == 'l':
        if state[0] - 1 < 1:
            new_state = state
        else:
            new_state = (state[0]-1, state[1])
    else:
        if state[0] + 1 > 4:
            new_state = state
        else:
            new_state = (state[0]+1, state[1])
    return new_state

def max_utility():
    
    return


def value_iteration(reward):
    '''
    Value iteration algorithm.

    Arguments
    ---------------
    reward: float
        specifes the reward for states
    Returns
    ---------------
    Optimal policy.
    '''
    # Initial utilities for each state
    utilities = {(1,1):0, (1,2):0, (1,3):0, (2,1):0, (2,2):None, (2,3):0, (3,1):0, (3,2):0, (3,3):0, (4,1):0, (4,2):0, (4,3):0}
    updated_utilites = {(1,1):0, (1,2):0, (1,3):0, (2,1):0, (2,2):None, (2,3):0, (3,1):0, (3,2):0, (3,3):0, (4,1):0, (4,2):0, (4,3):0}
    
    # Lists containing terminal and non terminal states
    non_terminal_states = list(utilities.keys())[:10]
    termimal_states = list(utilities.keys())[10:]
    all_states = list(utilities.keys())

    # The start state
    start_state = (1,1)

    epsilon = 0.01
    delta = 0
    discount_factor = 0.9

    flag = True
    
    while flag:
        utilies = updated_utilites
        delta = 0
        for state in all_states:
            possible_next_states = [transition_model(state = state, action = a) for a in ['u','l','d','r']]
            updated_utilites[state] = get_reward(state = state, reward = reward)

    return all_states

print(transition_model((1,3), 'u'))
