import pickle
import numpy as np

cluster_id = "radscan.563683."
job_numbers = 2424


def process_file(filename):
    file = open(filename, mode='r')
    lines = [line for line in file]
    if len(lines) < 1:
        return None
    # first 7 lines
    epsilon_k0 = float(lines[0].split(" ")[1])
    epsilon_k1 = float(lines[1].split(" ")[1])
    epsilon_k2 = float(lines[2].split(" ")[1])
    epsilon_k3 = float(lines[3].split(" ")[1])
    epsilon_k4 = float(lines[4].split(" ")[1])
    epsilon_k5 = float(lines[5].split(" ")[1])
    epsilon_k6 = float(lines[6].split(" ")[1])
    omega_x0 = float(lines[7].split(" ")[1])
    omega_y0 = float(lines[8].split(" ")[1])
    dx = float(lines[9].split(" ")[1])
    n_theta = int(lines[10].split(" ")[1])
    dtheta = float(lines[11].split(" ")[1])
    epsilon = float(lines[12].split(" ")[1])
    max_turns = float(lines[13].split(" ")[1])
    from_angle = float(lines[14].split(" ")[1])
    to_angle = float(lines[15].split(" ")[1])
    # everything else
    dictionary = {}
    for i in range(16, len(lines)):
        splitted = lines[i].split(" ")
        dictionary[float(splitted[0])] = np.asarray(
            [int(float(splitted[i])) for i in range(1,
                                                    len(splitted) - 1)],
            dtype=np.int)
    #print(dictionary)
    return (epsilon_k0, epsilon_k1, epsilon_k2, epsilon_k3, epsilon_k4,
            epsilon_k5, epsilon_k6, omega_x0, omega_y0, dx, n_theta, epsilon,
            max_turns, from_angle, to_angle, dictionary)

discarded = []

jobs = []

for i in range(job_numbers):
    print("Processing Job {}.".format(i))
    temp = process_file(cluster_id + str(i) + ".out")
    if temp is not None:
        jobs.append(temp)

dictionary = {}

for i in range(len(jobs)):
    if jobs[i][11] not in discarded:
        if (jobs[i][7], jobs[i][8], jobs[i][11]) in dictionary:
            dictionary[(jobs[i][7], jobs[i][8], jobs[i][11])] = {
                **dictionary[(jobs[i][7], jobs[i][8], jobs[i][11])],
                **jobs[i][15]
            }
        else:
            dictionary[(jobs[i][7], jobs[i][8], jobs[i][11])] = jobs[i][15]

#print(dictionary[1])

with open("revised_version_hennon.pkl", "wb") as f:
    pickle.dump(dictionary, f, pickle.HIGHEST_PROTOCOL)