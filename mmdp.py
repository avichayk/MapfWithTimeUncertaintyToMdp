import numpy as np
import mdptoolbox
import time

class mmdp_runner:
    ####################################################################################
    #  BUILD INPUT PARAMETERS

    FEE = -10000

    # board with security fence - real board indices starts at 1
    board = [
        [1,1,1,1,1],
        [1,0,0,0,1],
        [1,0,0,0,1],
        [1,0,0,0,1],
        [1,1,1,1,1]
    ]
    # adding an obstacle requires corrections at time_uncertainty per each action at locations surrounding the obstacle

    # agents' locations (starts from 1 - not 0)
    # assumption - starting points are not equal, agents don't disappear when reaching their goal
    agents = [ # [start_location, end_location]
        [[1, 1], [2, 3]],
        [[3, 1], [3, 3]]
    ]
    board[agents[0][0][0]][agents[0][0][1]] = 7
    board[agents[1][0][0]][agents[1][0][1]] = 7

    # time uncertainty from each location applying movement actions --  up, right, down, left, stay : in units
    time_uncertainty = [
        [ #up
            [FEE,FEE],  [FEE,FEE],  [FEE,FEE],  [FEE,FEE],  [FEE,FEE],
            [FEE,FEE],  [FEE,FEE],  [FEE,FEE],  [FEE,FEE],  [FEE,FEE],
            [FEE,FEE],  [1, 3],     [1, 3],     [1, 3],     [FEE,FEE],
            [FEE,FEE],  [1, 3],     [1, 3],     [1, 3],     [FEE,FEE],
            [FEE,FEE],  [FEE,FEE],  [FEE,FEE],  [FEE,FEE],  [FEE,FEE]
        ],
        [ #right
            [FEE,FEE],  [FEE,FEE],  [FEE,FEE],  [FEE,FEE],  [FEE,FEE],
            [FEE,FEE],  [1, 3],     [1, 3],     [FEE,FEE],  [FEE,FEE],
            [FEE,FEE],  [1, 3],     [1, 3],     [FEE,FEE],  [FEE,FEE],
            [FEE,FEE],  [1, 3],     [1, 3],     [FEE,FEE],  [FEE,FEE],
            [FEE,FEE],  [FEE,FEE],  [FEE,FEE],  [FEE,FEE],  [FEE,FEE]
        ],
        [ #down
            [FEE,FEE],  [FEE,FEE],  [FEE,FEE],  [FEE,FEE],  [FEE,FEE],
            [FEE,FEE],  [1, 3],     [1, 3],     [1, 3],     [FEE,FEE],
            [FEE,FEE],  [1, 3],     [1, 3],     [1, 3],     [FEE,FEE],
            [FEE,FEE],  [FEE,FEE],  [FEE,FEE],  [FEE,FEE],  [FEE,FEE],
            [FEE,FEE],  [FEE,FEE],  [FEE,FEE],  [FEE,FEE],  [FEE,FEE]
        ],
        [ #left
            [FEE,FEE],  [FEE,FEE],  [FEE,FEE],  [FEE,FEE],  [FEE,FEE],
            [FEE,FEE],  [FEE,FEE],  [1, 3],     [1, 3],     [FEE,FEE],
            [FEE,FEE],  [FEE,FEE],  [1, 3],     [1, 3],     [FEE,FEE],
            [FEE,FEE],  [FEE,FEE],  [1, 3],     [1, 3],     [FEE,FEE],
            [FEE,FEE],  [FEE,FEE],  [FEE,FEE],  [FEE,FEE],  [FEE,FEE]
        ],
        [ #stay
            [FEE,FEE],  [FEE,FEE],  [FEE,FEE],  [FEE,FEE],  [FEE,FEE],
            [FEE,FEE],  [1, 1],     [1, 1],     [1, 1],     [FEE,FEE],
            [FEE,FEE],  [1, 1],     [1, 1],     [1, 1],     [FEE,FEE],
            [FEE,FEE],  [1, 1],     [1, 1],     [1, 1],     [FEE,FEE],
            [FEE,FEE],  [FEE,FEE],  [FEE,FEE],  [FEE,FEE],  [FEE,FEE]
        ]
    ]

    # all relative movements - up, right, down, left, stay
    actions = [[-1,0], [0,1], [1,0], [0,-1], [0,0]]

    # states per agent -- start at starting point and append time starting at t=0 at columns
    S = [ # when adding agents -- add initial states HERE also
         [agents[0][0]],
         [agents[1][0]]
    ]
    S[0][0] = np.append(S[0][0], [0], 0)
    S[1][0] = np.append(S[1][0], [1], 0)

    states = np.array([S[0][0], S[1][0]])
    #agent_curr_state = [0, 1]  # updated during running

    start = time.time()

    # P=A*S*S, R=S*A duplicated by agents

    # init R per agent
    # -------------------
    R = [None] * len(agents)
    for r_agent_ind in range(len(agents)):
        R[r_agent_ind] = np.zeros((len(S[r_agent_ind]), len(actions)), dtype=object)

        for r_state_id in range(len(S[r_agent_ind])):
            for r_action_id in range(len(actions)):
                # check if action in state gets to goal

                if ((S[r_agent_ind][r_state_id][0] + actions[r_action_id][0]) == agents[r_agent_ind][1][0] and
                     (S[r_agent_ind][r_state_id][1] + actions[r_action_id][1]) == agents[r_agent_ind][1][1]):
                    # reward TO GOAL = 100
                    R[r_agent_ind][r_state_id][r_action_id] = 100
                else:
                    # reward (not to goal) = -lower_value_of_uncertainty
                    R[r_agent_ind][r_state_id][r_action_id] = \
                        (time_uncertainty[r_action_id][S[r_agent_ind][r_state_id][0] * 5 + S[r_agent_ind][r_state_id][1]])[0]  # 5 = width of board (update if board size changes)
    #R = np.array(R)

    # init P per agent
    # ignore uncertainty in normal transitions in initiation (the uncertainty of time is modeled by future states addition that also influences P)
    P = [None] * len(agents)
    for p_agent_ind in range(len(agents)):
        P[p_agent_ind] = [None] * len(actions)
        for p_action_id in range(len(actions)):
            P[p_agent_ind][p_action_id] = [None] * len(states)
            for p_state_id_outer in range(len(states)):
                P[p_agent_ind][p_action_id][p_state_id_outer] = np.array(np.empty(len(states)), dtype=object)
                for p_state_id_inner in range(len(states)):
                    P[p_agent_ind][p_action_id][p_state_id_outer][p_state_id_inner] = 0
                P[p_agent_ind][p_action_id][p_state_id_outer][p_state_id_outer] = 1

    P = np.array(P)

    ####################################################################################
    #  PLAN POLICY

    MAX_ITERATION = 5

    states_documentation = np.zeros((len(agents), MAX_ITERATION*10, len(actions))) # for time ranges uncertainty - iterate starting the documented state
    policies = []

    # TODO: handle ranges when adding states

    # open all possible states from starting point -- each agent separately not simultaneously
    for agent_ind in range(len(agents)):
        min_s = 0
        max_s = len(S[agent_ind])
        #curr_t = 0
        for iteration_ind in range(MAX_ITERATION):
            for state_added_ind in range(min_s, max_s):
                print(str(iteration_ind) + " " +str(state_added_ind) + " " + str(min_s) + " " + str(max_s))
                for action_ind in range(len(actions)):  # if not WALL and * range options
                    added_states = 0

                    max_range = (time_uncertainty[action_ind][S[agent_ind][state_added_ind][0] * 5 + S[agent_ind][state_added_ind][1]])[1]
                    min_range = (time_uncertainty[action_ind][S[agent_ind][state_added_ind][0] * 5 + S[agent_ind][state_added_ind][1]])[0]
                    old_t = S[agent_ind][state_added_ind][2]

                    # add possible states from initial states
                    if (board[S[agent_ind][state_added_ind][0] + actions[action_ind][0]][S[agent_ind][state_added_ind][1] + actions[action_ind][1]] != 1): # dont go to a wall...
                        added_states = max_range - min_range + 1

                        for ind_state_add in range(min_range, max_range + 1):
                            states = np.append(states, [[S[agent_ind][state_added_ind][0] + actions[action_ind][0],
                                                         S[agent_ind][state_added_ind][1] + actions[action_ind][1],
                                                         S[agent_ind][state_added_ind][2] + ind_state_add]], 0)  # not essential here
                            S[agent_ind] = np.append(S[agent_ind], [[S[agent_ind][state_added_ind][0] + actions[action_ind][0],
                                                                     S[agent_ind][state_added_ind][1] + actions[action_ind][1],
                                                                     S[agent_ind][state_added_ind][2] + ind_state_add]], 0)
                            if ind_state_add == min_range:
                                states_documentation[agent_ind][iteration_ind][action_ind] = len(S[agent_ind]) - 1 # document minimal path: change later?

                        # Update R
                        R_new = [None] * len(agents)
                        #for r_agent_id in range(len(agents)):

                        R_new[agent_ind] = np.zeros((len(S[agent_ind]), len(actions)), dtype=object)
                        for r_state_id in range(len(S[agent_ind]) - added_states):
                            for r_action_id in range(len(actions)):
                                R_new[agent_ind][r_state_id][r_action_id] = R[agent_ind][r_state_id][r_action_id]
                        for r_state_id in range(len(S[agent_ind]) - added_states, len(S[agent_ind])):
                            for r_action_id in range(len(actions)):  # for the NEW STATE
                                if ((S[agent_ind][r_state_id][0] + actions[r_action_id][0]) == agents[r_agent_ind][1][0] and
                                        (S[agent_ind][r_state_id][1] + actions[r_action_id][1]) == agents[r_agent_ind][1][1]):
                                    # reward TO GOAL = 100
                                    R_new[agent_ind][r_state_id][r_action_id] = 100
                                else:
                                    # reward (not to goal) = -lower_value_of_uncertainty
                                    print(time_uncertainty[r_action_id][S[agent_ind][r_state_id][0] * 5 + S[agent_ind][r_state_id][1]])
                                    val = (time_uncertainty[r_action_id][S[agent_ind][r_state_id][0] * 5 + S[agent_ind][r_state_id][1]])[0]  # 5 = width of board (update if board size changes)
                                    R_new[agent_ind][r_state_id][r_action_id] = val
                        R = R_new

                        old_state = state_added_ind
                        new_state = len(S[agent_ind]) - 1

                        # update P - transition allowed only to t+1 valid states created here
                        P_new = np.zeros((len(agents), len(actions), len(S[agent_ind]), len(S[agent_ind])), dtype=object)
                        #for p_agent_ind in range(len(agents)):
                        for p_action_id in range(len(actions)):
                            for p_state_id_outer in range(len(S[agent_ind]) - added_states):
                                for p_state_id_inner in range(len(S[agent_ind]) - added_states):
                                    P_new[agent_ind][p_action_id][p_state_id_outer][p_state_id_inner] = \
                                        P[agent_ind][p_action_id][p_state_id_outer][p_state_id_inner]
                            for i_p_states_added in range(1, added_states  + 1):
                                P_new[agent_ind][p_action_id][old_state][len(S[agent_ind]) - i_p_states_added] = 1 / added_states
                        P = P_new
            min_s = max_s
            max_s = len(S[agent_ind])
            #curr_t += 1
        # find policy
        vi = mdptoolbox.mdp.ValueIteration(P[agent_ind], R[agent_ind], discount=0.9, epsilon=0.01)
        vi.run()
        # next_action = vi.policy[loop_curr_state]
        policies = np.append(policies, [vi.policy])


    ####################################################################################
    #  SIMULATE
    curr_t = 0
    curr_iteration = 0
    completed = False
    next_state_ind = np.zeros(len(agents), dtype=int)

    while (not completed and curr_iteration < MAX_ITERATION):
        completed = True

        # make progress for agent (agent_ind) :: OPERATOR DECOMPOSITION
        for agent_ind in range(len(agents)):
            state = S[agent_ind][next_state_ind[agent_ind]]
            if (state[0] != agents[agent_ind][1][0] or
                    state[1] != agents[agent_ind][1][1]):
                completed = False # one of the agents did not reach destination

                next_state_ind[agent_ind] = states_documentation[agent_ind][curr_iteration][int(policies[agent_ind])]

                #
                calculate_x = state[0]
                calculate_y = state[1]
                action = int(policies[agent_ind])

                print("Agent: " + str(agent_ind) + " X: " + str(calculate_x) + ", Y: " + str(calculate_y) + ", A: " + str(action) + ", T: " + str(curr_iteration))
                print("Agent: " + str(agent_ind) + " X: " + str(calculate_x) + ", Y: " + str(calculate_y) + ", A: " + str(action) + ", T: " + str(state[2]))
                print("")

                # # update R of other agents - agent_ind moved
                # for other_agent_ind in range(len(agents)):
                #     if (other_agent_ind != agent_ind):
                #         for r_u_action in range(len(actions)):
                #             action_to_rev = actions[r_u_action]
                #             if (board[states[loop_curr_state][0] - action_to_rev[0]][states[loop_curr_state][1] - action_to_rev[1]] != 1):
                #                 for other_state in range(len(states)):
                #                     # old location no longer a obstacle - path to old location is allowed
                #                     if (states[other_state][0] + action_to_rev[0] == states[loop_curr_state][0] and
                #                         states[other_state][1] + action_to_rev[1] == states[loop_curr_state][1] and
                #                         board[states[other_state][0]][states[other_state][1]] != 1):
                #
                #                         R[other_agent_ind][other_state][r_u_action] = \
                #                                             (time_uncertainty[r_u_action][states[other_state][0] * 5 +
                #                                                    states[other_state][1]])[0] * -1  # 5 = width of board (update if board size changes)
                #                     # new location is an obstacle
                #                     if (states[other_state][0] + action_to_rev[0] == states[loop_curr_state + 1][0] and
                #                         states[other_state][1] + action_to_rev[1] == states[loop_curr_state + 1][1] and
                #                         board[states[other_state][0] + action_to_rev[0]][states[other_state][1] + action_to_rev[1]] != 1):
                #                             R[other_agent_ind][other_state][r_u_action] = MAX_ITERATION
            else:
                print("COMPLETED")
        curr_iteration += 1 # to stop at max iteration

    end = time.time()

    print("Time >> %.2f" % ((end - start) / 60))

    def printPolicy(policy , size=3):
        p = np.array(policy).reshape(size, size)
        range_F = range(size)
        print("    " + " ".join("%2d" % f for f in range_F))
        print("    " + "---" * size)
        for x in range(size):
            print(" %2d|" % x + " ".join("%2d" % p[x, f] for f in range_F))
