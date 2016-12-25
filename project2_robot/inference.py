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
	estimated_hidden_states = [None] * num_time_steps # remove this
	estimated_best = [None] * num_time_steps
	estimated_second = [None] * num_time_steps

	# to hold my forward messages
	messages_best = {}
	messages_second = {}
	messages_best[0] = {}
	messages_second[0] = {}
	pre_processing = prior_distribution
	for i in pre_processing:
		messages_best[0][i] = 0
		messages_second[0][i] = 0

	divergence = {}
	diverged = False

	# to hold my traceback messages
	traceback_best = {}
	traceback_second = {}

	initials_best = {}
	initials_second = {}

	###################
	# Store needed vars
	###################

	for i in range(1, num_time_steps):
	    messages_best[i] = {}
	    traceback_best[i-1] = {}
	    messages_second[i] = {}
	    traceback_second[i-1] = {}

	    initials_best[i] = {}
	    initials_second[i] = {}

	    divergence[i-1] = {}

	    # Initial adding of common elements from the observation and from the last
	    # forward message
	    if (observations[i-1] != None):
	        current_obs = log_post_observed[observations[i-1]]
	        for tup in current_obs:
	            key_a = [a for a, b in messages_best[i-1].items() if a[0:2] == tup]
	            key_b = [a for a, b in messages_second[i-1].items() if a[0:2] == tup]
	            if (key_a != []):
	                for key in key_a:
	                    initials_best[i][key] = current_obs[tup] + messages_best[i-1][key]
	            if (key_b != []):
	                for key in key_b:
	                    initials_second[i][key] = current_obs[tup] + messages_second[i-1][key]
	    		
	    else:
	        initials_best[i] = messages_best[i-1]
	        initials_second[i] = messages_second[i-1]

	    messages_best[i], traceback_best[i-1] = find_second(initials_best[i], second=False)

	    if diverged:
	    	messages_second[i], traceback_second[i-1] = find_second(initials_second[i], second=False)
	    else:
	    	messages_second[i], traceback_second[i-1] = find_second(initials_second[i], second=True)

	    for key in traceback_second[i-1]:
	    	if (key in traceback_best[i-1]):
	    		if (traceback_best[i-1][key] != traceback_second[i-1][key]):
	    			divergence[i-1][key] = True
	    			diverged = True
	    		else:
	    			divergence[i-1][key] = False
	    			diverged = False
	    	else:
	    		divergence[i-1][key] = True
	    		diverged = True


	# Just for the final node of the best estimate
	best_min_value = 1000000
	second_min_value = best_min_value
	best_min_key = None
	second_min_key = None
	obs = log_post_observed[observations[num_time_steps - 1]]
	temp_thingy = {}
	for nums in obs:
	    key_a = [a for a, b in messages_best[num_time_steps - 1].items() if a[0:2] == nums]
	    if key_a != []:
	        for key in key_a:
	            temp_thingy[key] = messages_best[num_time_steps - 1][key] + obs[nums]

	for key in temp_thingy:
	    if messages_best[num_time_steps - 1][key] < best_min_value:   
	        if (best_min_key == None):
	        	best_min_key = key
	        	second_min_key = key
	        	best_min_value = messages_best[num_time_steps - 1][key]
	        	second_min_value = messages_best[num_time_steps - 1][key]
	        else:
	        	second_min_value = best_min_value
	        	second_min_key = best_min_key
	        	best_min_value = messages_best[num_time_steps - 1][key]
	        	best_min_key = key


	estimated_best[num_time_steps - 1] = best_min_key
	estimated_second[num_time_steps - 1] = second_min_key


	different_path = False
	canceled = False

	potential_second_bests = [None] * num_time_steps
	possibilities = 0

	prob_tracker = [0] * num_time_steps
	best_prob = 0

	for i in reversed(range(num_time_steps - 1)):
		estimated_best[i] = traceback_best[i][estimated_best[i+1]]
		best_prob += messages_best[i+1][estimated_best[i+1]]

	# Compute estimated hidden states
	for j in range(num_time_steps):
		# potential_second_bests[possibilities] = [None] * num_time_steps
		estimated_second = [None] * num_time_steps
		estimated_second[num_time_steps - 1] = second_min_key
		temp_prob = 0
		for i in reversed(range(num_time_steps - 1)):
			if canceled == False:
				if (i == j or different_path == True):
					if i == j:
						if (divergence[i][estimated_second[i+1]] == False):
							canceled = True
						else:
							estimated_second[i] = traceback_second[i][estimated_second[i+1]]
							temp_prob += messages_second[i+1][estimated_second[i+1]]
							different_path = True
					else:
						if (divergence[i][estimated_second[i+1]] == True):
							estimated_second[i] = traceback_second[i][estimated_second[i+1]]
							temp_prob += messages_second[i+1][estimated_second[i+1]]
						else:
							estimated_second[i] = traceback_second[i][estimated_second[i+1]]
							temp_prob += messages_second[i+1][estimated_second[i+1]]
							different_path = False
				else:
					estimated_second[i] = traceback_best[i][estimated_second[i+1]]
					temp_prob += messages_second[i+1][estimated_second[i+1]]
		if canceled:
			canceled = False
		else:
			potential_second_bests[possibilities] = estimated_second
			prob_tracker[possibilities] = temp_prob
			possibilities += 1

	# Still have to evaluate them to find the best
	# Check until you get to None then break


	#estimated_hidden_states = estimated_best_extra

	min_score = 1000
	min_list = None
	continue_check = True

	for guess in potential_second_bests:
		if guess == None:
			break
		score = 0
		for location in range(len(guess)):
			if estimated_best[location] != guess[location]:
				score += 1
		print (score)
		if score < min_score and score != 0:
			min_score = score
			min_list = guess

	min_prob = 1000000
	min_prob_index = None

	for index in range(len(prob_tracker)):
		if prob_tracker[index] != 0 and prob_tracker[index] != best_prob:
			if prob_tracker[index] < min_prob:
				min_prob_index = index
				min_prob = prob_tracker[index]

	#estimated_hidden_states = min_list
	estimated_hidden_states = potential_second_bests[min_prob_index]

	print (prob_tracker)



	# # Compute estimated hidden states
	# for i in reversed(range(num_time_steps - 1)):
	#     #second_estimate[i] = new_traceback[i][second_estimate[i+1]]   
	#     if second_estimate[i+1] in traceback[i]:
	#         if second_estimate[i+1] in new_traceback[i]:
	#             if new_messages[i+1][second_estimate[i+1]] < messages[i+1][second_estimate[i+1]]:
	#                 second_estimate[i] = new_traceback[i][second_estimate[i+1]]
	#             else:
	#                 second_estimate[i] = traceback[i][second_estimate[i+1]]
	#         else:
	#             second_estimate[i] = traceback[i][second_estimate[i+1]]
	#     else:
	#         second_estimate[i] = new_traceback[i][second_estimate[i+1]]

	# estimated_hidden_states = second_estimate


	return estimated_hidden_states


# -----------------------------------------------------------------------------
# Generating data from the hidden Markov model
#

def find_second(initials, second=True):
	
	# This will hold the possible states of node+1
	potentials = {}
	output_messages = {}
	output_trace = {}

	if second:
		# Find out what the potential values of Xi are
	    for element in initials:
	        potentials.update(log_transition[element])

	    # Compute forward message values
	    for element in potentials:
	    	# This will hold all possible node locations that can 
	    	# generate a node+1 location in potentials
	        temp_list = log_post_transition[element]
	        min_value = 1000000.0
	        second_min_value = min_value
	        min_key = None
	        second_min_key = None
	        min_dict = {}

	        # Fill min_dict with the correct values of the plausible
	        # locations in temp_list
	        for k in temp_list:
	            if (k in initials):
	                if (k not in min_dict):
	                    min_dict[k] = initials[k] + temp_list[k]
	                else:
	                    min_dict[k] += initials[k] + temp_list[k]

	        for k in min_dict:
	            if ((k != None) & (min_dict[k] < min_value)):
		            if (second_min_key == None):
		                second_min_value = min_dict[k]
		                second_min_key = k
		            else:
		                second_min_value = min_value
		                second_min_key = min_key
		            min_value = min_dict[k]
		            min_key = k

	        if (min_key != None):
	            output_messages[element] = second_min_value
	            output_trace[element] = second_min_key
	    return output_messages, output_trace

	else:	
		# Find out what the potential values of Xi are
		for element in initials:
			potentials.update(log_transition[element])

        # Compute forward message values
		for element in potentials:
        	# This will hold all possible node locations that can 
        	# generate a node+1 location in potentials
			temp_list = log_post_transition[element]
			min_value = 1000000.0
			min_key = None
			min_dict = {}

            # Fill min_dict with the correct values of the plausible
            # locations in temp_list
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
			    output_messages[element] = min_value
			    output_trace[element] = min_key

		# return dictionary of second-best messages and
		# corresponding dictionary of traceback messages
		return output_messages, output_trace



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
