import numpy as np
import mdptoolbox
import time

class mmdp_runner:
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

    # TODO: update R in initiation with agents as obstacles to other agents

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
    agent_curr_state = [0, 1]  # updated during running
    print(states)
    start = time.time()

    # P=A*S*S, R=S*A duplicated by agents

    # init R per agent
    # -------------------
    R = np.zeros((len(agents), curr_s, len(actions)))
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
    P = np.zeros((len(agents), len(actions), curr_s, curr_s))  # TODO: change P to have no  uncertainty in initial states

    # ignore uncertainty in normal transitions in initiation (the uncertainty of time is modeled by future states addition that also influences P)
    for p_agent_ind in range(len(agents)):
        for p_action_id in range(len(actions)):
            for p_state_id_outer in range(curr_s):
                for p_state_id_inner in range(curr_s):
                    if (p_state_id_inner == p_state_id_outer):
                        P[p_agent_ind][p_action_id][p_state_id_outer][p_state_id_inner] = 1

    MAX_ITERATION = 10000
    curr_iteration = 0
    completed = False

    while (not completed and curr_iteration < MAX_ITERATION):
        completed = True

        # make progress for agent (agent_ind) :: OPERATOR DECOMPOSITION
        for agent_ind in range(len(agents)):
            if (S[agent_ind][-1][0] != agents[agent_ind][1][0] or S[agent_ind][-1][1] != agents[agent_ind][1][1]):
                completed = False
                loop_curr_state = agent_curr_state[agent_ind]

                # find policy
                vi = mdptoolbox.mdp.ValueIteration(P[agent_ind], R[agent_ind], discount=0.9, epsilon=0.01)
                vi.run()
                next_action = vi.policy[loop_curr_state]


                states = np.append(states, [[states[loop_curr_state][0] + actions[next_action][0],
                                                     states[loop_curr_state][1] + actions[next_action][1],
                                                     curr_t]], 0)
                S[agent_ind] = np.append(S[agent_ind], [[states[loop_curr_state][0] + actions[next_action][0],
                                                                 states[loop_curr_state][1] + actions[next_action][1],
                                                                 curr_t]], 0)
                R[agent_ind] = np.append(R[agent_ind], [np.zeros(len(actions))], 0) # TODO: add col to R...

                for r_action_id in range(len(actions)):
                    # check if action in state gets to goal

                    if ((states[-1][0] + actions[r_action_id][0]) == agents[r_agent_ind][1][0] and
                            (states[-1][1] + actions[r_action_id][1]) == agents[r_agent_ind][1][1]):
                        # reward TO GOAL = 100
                        R[r_agent_ind][-1][r_action_id] = 100
                    else:
                        # reward (not to goal) = -lower_value_of_uncertainty
                        R[r_agent_ind][r_state_id][r_action_id] = \
                            (time_uncertainty[r_action_id][states[-1][0] * 5 +
                                                           states[-1][1]])[
                                0] * -1  # 5 = width of board (update if board size changes)

                print(R[agent_ind])

                agent_curr_state[agent_ind] = len(states)

                # update P

                # update R of other agents - agent_ind moved
                for other_agent_ind in range(len(agents)):
                    if (other_agent_ind != agent_curr_state):
                        for r_u_action in range(len(actions)):
                            rev_action = actions[r_u_action]
                            if (board[states[loop_curr_state][0] - rev_action[0]][states[loop_curr_state][1] - rev_action[1]] != 1):
                                for other_state in range(len(states)):
                                    # old location no longer a obstacle
                                    if (states[other_state][0] + rev_action[0] == states[loop_curr_state][0] and
                                        states[other_state][1] + rev_action[1] == states[loop_curr_state][1] and
                                        board[states[other_state][0] + rev_action[0]][states[other_state][1] + rev_action[1] != 1]):
                                            R[other_agent_ind][other_state][r_u_action] = \
                                                                (time_uncertainty[r_u_action][states[other_state][0] * 5 +
                                                                       states[other_state][1]])[0] * -1  # 5 = width of board (update if board size changes)
                                    # new location is an obstacle
                                    if (states[other_state][0] + actions[r_u_action][0] == states[loop_curr_state+1][0] and
                                        states[other_state][1] + actions[r_u_action][1] == states[loop_curr_state+1][1] and
                                        board[states[other_state][0] + actions[r_u_action][0]][states[other_state][1] + actions[r_u_action][1] != 1]):
                                            R[other_agent_ind][other_state][r_u_action] = MAX_ITERATION  # 5 = width of board (update if board size changes)

        curr_t += 1
        curr_iteration += 1

    end = time.time()

    print("%.2f" % ((end - start) / 60))


    def printPolicy(policy , size=3):
        p = np.array(policy).reshape(size, size)
        range_F = range(size)
        print("    " + " ".join("%2d" % f for f in range_F))
        print("    " + "---" * size)
        for x in range(size):
            print(" %2d|" % x + " ".join("%2d" % p[x, f] for f in range_F))
