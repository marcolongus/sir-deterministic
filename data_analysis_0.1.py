import numpy as np
import matplotlib.pyplot as plt
from more_itertools import chunked


def outbreak_detector(simulation: np.array, n_batch=20):
	# simulation: batch of system evolution with the same seed
	# Store epidemic size at each entry. 
	outbreak = []
	outbreak_flag = False
	for i in range(n_batch-1):
		if simulation[i]<simulation[i+1]:
			outbreak.append(simulation[i+1]-simulation[i])
			outbreak_flag = True
	return [outbreak, outbreak_flag]

def get_threshold(size: np.array, threshold=50, n_simulations=10, n_batch=20):
    threshold_count = 0 
    for i in range(n_simulations):
        if size[i * n_batch : (i + 1) * n_batch][0] >= threshold:
            threshold_count += 1
    return threshold_count

def get_size_matrix(size, threshold=50, n_simulations=10, n_batch=20):
	size_matrix = np.zeros(shape=(get_threshold(size), n_batch))
	threshold_count = 0
	for i in range(n_simulations):
		simulation_i = size[i * n_batch : (i + 1) * n_batch]
		if simulation[0] >= threshold:
			size_matrix[threshold_count,:] = simulation_i
			threshold_count+=1
	return size_matrix

#==================================================================================#
# Main
#==================================================================================#

data = np.loadtxt("data/pl_noisy_new/evolution.txt")
size = data[:, 0]

for simulation in chunked(size, 20):
	print(outbreak_detector(simulation))

data_matrix = get_size_matrix(size)

x_imnues = np.arange(0, 200, 10)
x_imnues[0]= 10
x_imnues = x_imnues - 10

for i in range(10):
	plt.scatter(x_imnues, data_matrix[i,:])
	plt.plot(x_imnues, data_matrix[i,:])
	plt.show()