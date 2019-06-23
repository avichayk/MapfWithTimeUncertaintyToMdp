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

    # time uncertainty from each location applying movement actions --  up, right, down, left, stay
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

    curr_t = 1
    curr_s = len(agents) # so far 1 state added for each agent
    states = np.array([S[0][0], S[1][0]])
    #agent_curr_state = [0, 1]  # updated during running

    start = time.time()

    # P=A*S*S, R=S*A duplicated by agents

    # init R per agent
    # -------------------
    R = np.zeros((len(agents), curr_s, len(actions)), dtype=object)
    for r_agent_ind in range(len(agents)):
        for r_state_id in range(curr_s):
            for r_action_id in range(len(actions)):
                # check if action in state gets to goal

                if ((states[r_state_id][0] + actions[r_action_id][0]) == agents[r_agent_ind][1][0] and
                     (states[r_state_id][1] + actions[r_action_id][1]) == agents[r_agent_ind][1][1]):
                    # reward TO GOAL = 100
                    R[r_agent_ind][r_state_id][r_action_id] = 100
                else:
                    # reward (not to goal) = -lower_value_of_uncertainty
                    R[r_agent_ind][r_state_id][r_action_id] = \
                        (time_uncertainty[r_action_id][states[r_state_id][0] * 5 +
                                                       states[r_state_id][1]])[0] * -1  # 5 = width of board (update if board size changes)

    # init P per agent
    # ignore uncertainty in normal transitions in initiation (the uncertainty of time is modeled by future states addition that also influences P)
    P = [None] * len(agents)
    for p_agent_ind in range(len(agents)):
        P[p_agent_ind] = [None] * len(actions)
        for p_action_id in range(len(actions)):
            P[p_agent_ind][p_action_id] = [None] * curr_s
            for p_state_id_outer in range(curr_s):
                P[p_agent_ind][p_action_id][p_state_id_outer] = np.array(np.empty(curr_s), dtype=object)
                for p_state_id_inner in range(curr_s):
                    P[p_agent_ind][p_action_id][p_state_id_outer][p_state_id_inner] = 0
                P[p_agent_ind][p_action_id][p_state_id_outer][p_state_id_outer] = 1

    P = np.array(P)

    ####################################################################################
    #  PLAN POLICY

    MAX_ITERATION = 10

    states_documentation = np.zeros((len(agents), MAX_ITERATION, len(actions))) # for time ranges uncertainty - iterate starting the documented state
    policies = []

    # open all possible states from starting point
    for agent_ind in range(len(agents)):
        min_s = 0
        max_s = len(S[agent_ind])
        for iteration_ind in range(MAX_ITERATION):
            for state_add_ind in range(min_s, max_s):
                for action_ind in range(len(actions)):  # if not WALL and * range options
                    # add possible states from initial states
                    if (board[states[min_s][0] + actions[action_ind][0]][states[min_s][1] + actions[action_ind][1]] != 1):
                        states = np.append(states, [[states[min_s][0] + actions[action_ind][0],
                                                     states[min_s][1] + actions[action_ind][1],
                                                     curr_t + 1]], 0)  # not essential here
                        S[agent_ind] = np.append(S[agent_ind], [[states[min_s][0] + actions[action_ind][0],
                                                                 states[min_s][1] + actions[action_ind][1],
                                                                 curr_t + 1]], 0)

                        states_documentation[agent_ind][iteration_ind][action_ind] = len(S[agent_ind]) - 1

                        # Update R
                        R_new = np.zeros((len(agents), curr_s + 1, len(actions)), dtype=object)
                        for r_agent_id in range(len(agents)):
                            for r_state_id in range(curr_s):
                                for r_action_id in range(len(actions)):
                                    R_new[r_agent_id][r_state_id][r_action_id] = R[r_agent_id][r_state_id][r_action_id]
                            for r_action_id in range(len(actions)):  # for the NEW STATE
                                if ((states[curr_s][0] + actions[r_action_id][0]) == agents[r_agent_ind][1][0] and
                                        (states[curr_s][1] + actions[r_action_id][1]) == agents[r_agent_ind][1][1]):
                                    # reward TO GOAL = 100
                                    R_new[r_agent_ind][curr_s][r_action_id] = 100
                                else:
                                    # reward (not to goal) = -lower_value_of_uncertainty
                                    R_new[r_agent_ind][curr_s][r_action_id] = \
                                        (time_uncertainty[r_action_id][states[curr_s][0] * 5 +
                                                                       states[curr_s][1]])[
                                            0] * -1  # 5 = width of board (update if board size changes)
                        R = R_new

                        # update P
                        P_new = np.zeros((len(agents), len(actions), curr_s + 1, curr_s + 1), dtype=object)
                        for p_agent_ind in range(len(agents)):
                            for p_action_id in range(len(actions)):
                                for p_state_id_outer in range(curr_s):
                                    for p_state_id_inner in range(curr_s):
                                        P_new[p_agent_ind][p_action_id][p_state_id_outer][p_state_id_inner] = \
                                            P[p_agent_ind][p_action_id][p_state_id_outer][p_state_id_inner]
                                P_new[p_agent_ind][p_action_id][curr_s][curr_s] = 1
                        P = P_new
            min_s = max_s
            max_s = len(S[agent_ind])
            curr_t += 1

        # find policy
        vi = mdptoolbox.mdp.ValueIteration(P[agent_ind], R[agent_ind], discount=0.9, epsilon=0.01)
        vi.run()
        #next_action = vi.policy[loop_curr_state]
        policies = np.append(policies, [vi.policy])

    ####################################################################################
    #  SIMULATE
    curr_t = 0
    curr_iteration = 0
    completed = False
    next_state_ind = np.zeros(len(agents))

    while (not completed and curr_iteration < MAX_ITERATION):
        completed = True

        # make progress for agent (agent_ind) :: OPERATOR DECOMPOSITION
        for agent_ind in range(len(agents)):

            if (S[agent_ind][next_state_ind[agent_ind]][0] != agents[agent_ind][1][0] or S[agent_ind][next_state_ind[agent_ind]][1] != agents[agent_ind][1][1]):
                completed = False # one of the agents did not reach destination

                next_state_ind[agent_ind] = states_documentation[agent_ind][iteration_ind][S[agent_ind][iteration_ind]]

                calculate_x = S[agent_ind][next_state_ind][0]
                calculate_y = S[agent_ind][next_state_ind][1]

                print(S[agent_ind][next_state_ind])

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
        curr_iteration += 1 # to stop at max iteration

    end = time.time()

    print("%.2f" % ((end - start) / 60))

    def printPolicy(policy , size=3):
        p = np.array(policy).reshape(size, size)
        range_F = range(size)
        print("    " + " ".join("%2d" % f for f in range_F))
        print("    " + "---" * size)
        for x in range(size):
            print(" %2d|" % x + " ".join("%2d" % p[x, f] for f in range_F))
