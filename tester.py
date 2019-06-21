import mdptoolbox.example
import os.path
import csv
import sys
from pathfinding.map_reader import *
from pathfinding.conformant_cbs import *
from pathfinding.operator_decomposition_a_star import *
import numpy as np

def printPolicy(policy):
    p = np.array(policy).reshape(3, 3)
    range_F = range(3)
    print("    " + " ".join("%2d" % f for f in range_F))
    print("    " + "---" * 3)
    for x in range(3):
        print(" %2d|" % x + " ".join("%2d" % p[x, f] for f in range_F))

class Experiments:
    def __init__(self, output_folder):
        self.output_folder = output_folder
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        self.num_of_reps = 1
        self.max_agents_num = 2
        self.uncertainty = 0
        self.time_limit = 300
        self.file_prefix = 'default - '


    def run_blank_map(self, rep_num, agent_num):
        results_file = self.file_prefix + 'small_open_map_results.csv'
        map_file = '../maps/small_blank_map.map'
        seed = 12345678
        random.seed(seed)
        blank_problem = ConformantProblem(map_file)
        blank_problem.generate_problem_instance(self.uncertainty)
        print(f"--- STARTED BLANK MAP | SEED: {seed} | UNCERTAINTY: {self.uncertainty} ---")
        #self.run_and_log_same_instance_experiment(blank_problem, results_file, agent_num, rep_num, seed)

P1 = np.array([
    [[0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
     [0.10, 0.20, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20]],

    [[0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20]],

    [[0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20]],

    [[0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20],
     [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.20]]
])

R1 = np.array([
              [0, 2, 2, 2],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 2, 2, 2],
              [0, 0, 0, 0],
              [0, 0, 0, 0],
              [0, 0, 0, 20]
              ])

#R1C = np.zeros((9, 4))

start = time.time()

vi = mdptoolbox.mdp.ValueIteration(P1, R1, discount=0.9, epsilon=0.01)
vi.run()

end = time.time()


print("%.2f" % ((end - start) / 60))

vi.policy # result is (0, 0, 0)
print(vi.policy)
#printPolicy(vi.policy[:, 0])
printPolicy(vi.policy[:])


# run simulation -- random number chosen - is it upper than p


exp = Experiments('./../experiments')
#exp.run_experiments_on_same_instance(num_of_agents=2, uncertainty=1, time_limit=60, rep_num=5)

#print("Finished Experiments")


np.random.choice(5, 1, p=[0.1, 0, 0.3, 0.6, 0])


