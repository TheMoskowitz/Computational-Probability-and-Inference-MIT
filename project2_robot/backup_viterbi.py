#!/usr/bin/env python
# inference.py
# Base code by George H. Chen (georgehc@mit.edu) -- updated 10/18/2016
import collections
import sys

import graphics
import numpy as np
import robot


# Throughout the code, we use these variables.
# Do NOT change these (but you'll need to use them!):
# - all_possible_hidden_states: a list of possible hidden states
# - all_possible_observed_states: a list of possible observed states
# - prior_distribution: a distribution over states
# - transition_model: a function that takes a hidden state and returns a
#     Distribution for the next state
# - observation_model: a function that takes a hidden state and returns a
#     Distribution for the observation from that hidden state
all_possible_hidden_states = robot.get_all_hidden_states()
all_possible_observed_states = robot.get_all_observed_states()
prior_distribution = robot.initial_distribution()
transition_model = robot.transition_model
observation_model = robot.observation_model

observed_matrix = {}

for state in all_possible_observed_states:
    x,y = state
    new_tuple = (x,y,'stay')
    observed_matrix[state] = robot.Distribution()
    observed_matrix[state] = observation_model(new_tuple)

posterior_observed_matrix = {}

for i in observed_matrix:
    for j in observed_matrix[i]:
        if (j not in posterior_observed_matrix):
            posterior_observed_matrix[j] = robot.Distribution()
        posterior_observed_matrix[j][i] = observed_matrix[i][j]

transition_matrix = {}

for state in all_possible_hidden_states:
    transition_matrix[state] = robot.Distribution()
    transition_matrix[state] = transition_model(state)

# And now to transpose it
posterior_transition_matrix = {}

for i in transition_matrix:
    for j in transition_matrix[i]:
        if (j not in posterior_transition_matrix):
            posterior_transition_matrix[j] = robot.Distribution()
        posterior_transition_matrix[j][i] = transition_matrix[i][j]


# You may find this function helpful for computing logs without yielding a
# NumPy warning when taking the log of 0.
def careful_log(x):
    # computes the log of a non-negative real number
    if x == 0:
        return -np.inf
    else:
        return np.log(x)


# -----------------------------------------------------------------------------
# Functions for you to implement
#

def forward_backward(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of marginal distributions at each time step; each distribution
    should be encoded as a Distribution (see the Distribution class in
    robot.py and see how it is used in both robot.py and the function
    generate_data() above, and the i-th Distribution should correspond to time
    step i
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #

    num_time_steps = len(observations)
    forward_messages = [None] * num_time_steps
    forward_messages[0] = prior_distribution
    # TODO: Compute the forward messages


    for i in range(num_time_steps - 1):
        forward_messages[i+1] = robot.Distribution()
        if (observations[i] != None):
            current_obs = posterior_observed_matrix[observations[i]]

            dist_to_add = robot.Distribution()
            for tup in current_obs:
                key_a = [a for a, b in forward_messages[i].items() if a[0:2] == tup]
                if key_a != []:
                    for item in key_a:
                        new_prob = forward_messages[i][item] * current_obs[tup]
                        temp_dist = transition_model(item)
                        for k in temp_dist:
                            new_val = temp_dist[k] * new_prob
                            dist_to_add[k] += new_val
        else:
            dist_to_add = robot.Distribution()
            for tup in forward_messages[i]:
                temp_dist = transition_model(tup)
                for key in temp_dist:
                    dist_to_add[key] += temp_dist[key] * forward_messages[i][tup]

        forward_messages[i+1].update(dist_to_add)
        forward_messages[i+1].renormalize()



    #print ('complete')
    #print (forward_messages[3])
    backward_messages = [None] * num_time_steps
    backward_messages[-1] = robot.Distribution()

    for state in all_possible_hidden_states:
        backward_messages[-1][state] = 1.0 / len(all_possible_hidden_states)
    

    # Because there is no posterior_transition_model, I'm forced to 
    # either a) make one myself or b) fill out an entire transition
    # matrix just so I can transpose it. B is a tremendous waste of 
    # space but much faster to code, so I'm trying it.


    for i in reversed(range(num_time_steps - 1)):
        backward_messages[i] = robot.Distribution()
        if (observations[i+1] != None):
            next_obs = posterior_observed_matrix[observations[i+1]]

            dist_to_add = robot.Distribution()
            for tup in next_obs:
                key_a = [a for a, b in backward_messages[i+1].items() if a[0:2] == tup]
                if key_a != []:
                    for item in key_a:
                        new_prob = backward_messages[i+1][item] * next_obs[tup]
                        temp_dist = posterior_transition_matrix[item]
                        for k in temp_dist:
                            new_val = temp_dist[k] * new_prob
                            dist_to_add[k] += new_val
        else:
            dist_to_add = robot.Distribution()
            for tup in backward_messages[i+1]:
                temp_dist = posterior_transition_matrix[tup]
                for key in temp_dist:
                    dist_to_add[key] += temp_dist[key] * backward_messages[i+1][tup]

        backward_messages[i].update(dist_to_add)
        backward_messages[i].renormalize()

    #print (backward_messages[3])


    marginals = [None] * num_time_steps # remove this
    # TODO: Compute the marginals 
    for i in range(num_time_steps):

        marginals[i] = robot.Distribution()
        shared_keys = robot.Distribution()

        for key in forward_messages[i]:
            if (key in backward_messages[i]):
                shared_keys[key] = forward_messages[i][key] * backward_messages[i][key]
        
        if (observations[i] != None):
            current_obs = posterior_observed_matrix[observations[i]]
            for tup in current_obs:
                key_a = [a for a, b in shared_keys.items() if a[0:2] == tup]
                if key_a != []:
                    for item in key_a:
                        new_prob = current_obs[tup] * shared_keys[item]
                        marginals[i][item] = new_prob
        else:
            marginals[i].update(shared_keys)

        marginals[i].renormalize()
    return marginals


def Viterbi(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #


    num_time_steps = len(observations)
    estimated_hidden_states = [None] * num_time_steps # remove this

    log_post_observed = {}
    for i in posterior_observed_matrix:
        log_post_observed[i] = {}
        for j in posterior_observed_matrix[i]:
            log_post_observed[i][j] = -careful_log(posterior_observed_matrix[i][j])

    log_transition = {}
    for i in transition_matrix:
        log_transition[i] = {}
        for j in transition_matrix[i]:
            log_transition[i][j] = -careful_log(transition_matrix[i][j])

    log_post_transition = {}
    for i in posterior_transition_matrix:
        log_post_transition[i] = {}
        for j in posterior_transition_matrix[i]:
            log_post_transition[i][j] = -careful_log(posterior_transition_matrix[i][j])


    # Enough preamble setting up the log versions of my dictionary matrices
    # Now for the actual code!

    # to hold my forward messages
    messages = {}
    messages[0] = {}
    pre_processing = prior_distribution
    for i in pre_processing:
        #messages[0][i] = -careful_log(pre_processing[i])
        messages[0][i] = 0


    # to hold my traceback messages
    traceback = {}

    for i in range(1, num_time_steps):
        messages[i] = {}
        traceback[i-1] = {}

        initials = {}
        potentials = {}

        # Initial adding of common elements from the observation and from the last
        # forward message
        if (observations[i-1] != None):
            current_obs = log_post_observed[observations[i-1]]
            for tup in current_obs:
                key_a = [a for a, b in messages[i-1].items() if a[0:2] == tup]
                if (key_a != []):
                    for key in key_a:
                        initials[key] = current_obs[tup] + messages[i-1][key]
        else:
            initials = messages[i-1]

        # Find out what the potential values of Xi are
        for element in initials:
            potentials.update(log_transition[element])


        # Compute forward message values
        for element in potentials:
            temp_list = log_post_transition[element]
            min_value = 1000000.0
            min_key = None
            min_dict = {}

            for k in temp_list:
                if (k in initials):
                    if (k not in min_dict):
                        min_dict[k] = initials[k] + temp_list[k]
                    else:
                        min_dict[k] += initials[k] + temp_list[k]
            for k in min_dict:
                if ((k != None) & (min_dict[k] < min_value)):
                    min_value = min_dict[k]
                    min_key = k

            if (min_key != None):
                messages[i][element] = min_value
                traceback[i-1][element] = min_key


    # Just for the final node
    min_value = 1000000
    min_key = None
    obs = log_post_observed[observations[num_time_steps - 1]]
    temp_thingy = {}
    for nums in obs:
        key_a = [a for a, b in messages[num_time_steps - 1].items() if a[0:2] == nums]
        if key_a != []:
            for key in key_a:
                temp_thingy[key] = messages[num_time_steps - 1][key] + obs[nums]

    for key in temp_thingy:
        if messages[num_time_steps - 1][key] < min_value:
                min_value = messages[num_time_steps - 1][key]
                min_key = key
    estimated_hidden_states[num_time_steps -1] = min_key


    # Compute estimated hidden states
    for i in reversed(range(num_time_steps - 1)):
        estimated_hidden_states[i] = traceback[i][estimated_hidden_states[i+1]]

    return estimated_hidden_states


def second_best(observations):
    """
    Input
    -----
    observations: a list of observations, one per hidden state
        (a missing observation is encoded as None)

    Output
    ------
    A list of esimated hidden states, each encoded as a tuple
    (<x>, <y>, <action>)
    """

    # -------------------------------------------------------------------------
    # YOUR CODE GOES HERE
    #


    num_time_steps = len(observations)
    (
        possible_hidden_states,
        possible_previous_hidden_states
    ) = compute_possible_hidden_states()
    
    # Compute the forward max messages
    forward_max_messages = []
    
    observation = observations[0]
    
    message = compute_initial_forward_max_message(
        observation,
        possible_hidden_states
    )
    
    forward_max_messages.append(message)
    
    previous_message = message
    
    for time_step in range(1, num_time_steps):
        observation = observations[time_step]
        message = compute_next_forward_max_message(
            observation,
            previous_message,
            possible_hidden_states
        )
        forward_max_messages.append(message)
        previous_message = message
     
    # Compute the traceback
    estimated_hidden_states = [None] * num_time_steps
    message = forward_max_messages[-1]
    paths = []
    path_queue = []
    
    for hidden_state in message:
        node = (
            hidden_state,
            message[hidden_state][1]
        )
        path_queue.append([ node ])
                
    while len(path_queue) > 0:
        path = path_queue[0]
        path_queue = path_queue[1:]
        previous_time_step = num_time_steps - len(path) - 1        
        last_node = path[-1]
        previous_hidden_states = last_node[1]

        if len(previous_hidden_states) == 0:
            paths.append(path)            
        else:
            for previous_hidden_state in previous_hidden_states:
                message = forward_max_messages[previous_time_step]
                node = (
                    previous_hidden_state,
                    message[previous_hidden_state][1]
                )
                path_ = path.copy()
                path_.append(node)
                path_queue.append(path_)
    scored_paths = []
    
    for path in paths:
        score = 0.
        path = np.flipud(path)
        hidden_state = path[0][0]
        prob_observation = observation_model(hidden_state)[observations[0]]
        prob_prior = prior_distribution[hidden_state]
        
        if prob_observation > 0 and prob_prior > 0:
            score = np.log(prob_observation) + np.log(prob_prior)
        else:
            continue
            
        for index in range(1, num_time_steps):
            hidden_state = path[index][0]
            previous_hidden_state = path[index - 1][0]
            prob_observation = observation_model(hidden_state)[observations[index]]
            prob_transition = transition_model(previous_hidden_state)[hidden_state]
            if prob_observation > 0 and prob_transition > 0:
                score = score + np.log(prob_observation) + np.log(prob_transition)
            else:
                score = -np.inf
                break
            
        if score > -np.inf:
            scored_paths.append([ score, path ])
                    
    scored_paths = sorted(scored_paths, key=lambda x: -x[0])
    second_best_path = scored_paths[1][1]
    estimated_hidden_states = list(map(lambda x: x[0], second_best_path))
    
    return estimated_hidden_states

# -----------------------------------------------------------------------------
# Generating data from the hidden Markov model
#

def generate_data(num_time_steps, make_some_observations_missing=False,
                  random_seed=None):
    # generate samples from this project's hidden Markov model
    hidden_states = []
    observations = []

    # if the random seed is not None, then this makes the randomness
    # deterministic, which may be helpful for debug purposes
    np.random.seed(random_seed)

    # draw initial state and emit an observation
    initial_state = prior_distribution.sample()
    initial_observation = observation_model(initial_state).sample()

    hidden_states.append(initial_state)
    observations.append(initial_observation)

    for time_step in range(1, num_time_steps):
        # move the robot
        prev_state = hidden_states[-1]
        new_state = transition_model(prev_state).sample()

        # maybe emit an observation
        if not make_some_observations_missing:
            new_observation = observation_model(new_state).sample()
        else:
            if np.random.rand() < .1:  # 0.1 prob. of observation being missing
                new_observation = None
            else:
                new_observation = observation_model(new_state).sample()

        hidden_states.append(new_state)
        observations.append(new_observation)

    return hidden_states, observations


# -----------------------------------------------------------------------------
# Main
#

def main():
    # flags
    make_some_observations_missing = False
    use_graphics = True
    need_to_generate_data = True

    # parse command line arguments
    for arg in sys.argv[1:]:
        if arg == '--missing':
            make_some_observations_missing = True
        elif arg == '--nographics':
            use_graphics = False
        elif arg.startswith('--load='):
            filename = arg[7:]
            hidden_states, observations = robot.load_data(filename)
            need_to_generate_data = False
            num_time_steps = len(hidden_states)

    # if no data is loaded, then generate new data
    if need_to_generate_data:
        num_time_steps = 100
        hidden_states, observations = \
            generate_data(num_time_steps,
                          make_some_observations_missing)

    print('Running forward-backward...')
    marginals = forward_backward(observations)
    print("\n")

    timestep = 2
    print("Most likely parts of marginal at time %d:" % (timestep))
    if marginals[timestep] is not None:
        print(sorted(marginals[timestep].items(),
                     key=lambda x: x[1],
                     reverse=True)[:10])
    else:
        print('*No marginal computed*')
    print("\n")

    print('Running Viterbi...')
    estimated_states = Viterbi(observations)
    print("\n")

    print("Last 10 hidden states in the MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states[time_step] is None:
            print('Missing')
        else:
            print(estimated_states[time_step])
    print("\n")

    print('Finding second-best MAP estimate...')
    estimated_states2 = second_best(observations)
    print("\n")

    print("Last 10 hidden states in the second-best MAP estimate:")
    for time_step in range(num_time_steps - 10 - 1, num_time_steps):
        if estimated_states2[time_step] is None:
            print('Missing')
        else:
            print(estimated_states2[time_step])
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP estimate and true hidden " +
          "states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states2[time_step] != hidden_states[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between second-best MAP estimate and " +
          "true hidden states:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    difference = 0
    difference_time_steps = []
    for time_step in range(num_time_steps):
        if estimated_states[time_step] != estimated_states2[time_step]:
            difference += 1
            difference_time_steps.append(time_step)
    print("Number of differences between MAP and second-best MAP " +
          "estimates:", difference)
    if difference > 0:
        print("Differences are at the following time steps: " +
              ", ".join(["%d" % time_step
                         for time_step in difference_time_steps]))
    print("\n")

    # display
    if use_graphics:
        app = graphics.playback_positions(hidden_states,
                                          observations,
                                          estimated_states,
                                          marginals)
        app.mainloop()


if __name__ == '__main__':
    main()
