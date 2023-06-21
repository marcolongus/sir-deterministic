import numpy as np
import matplotlib.pyplot as plt
from more_itertools import chunked


def outbreak_detector(simulation: np.array, n_batch=50):
	# simulation: batch of system evolution with the same seed
	# Store epidemic size at each entry. 
	outbreak = []
	outbreak_flag = False
	for i in range(n_batch-1):
		if simulation[i]<simulation[i+1]:
			outbreak.append(simulation[i+1]-simulation[i])
			outbreak_flag = True
	return [outbreak, outbreak_flag]

def get_threshold(size: np.array, threshold=0, n_simulations=29, n_batch=50):
    threshold_count = 0 
    for i in range(n_simulations):
        if size[i * n_batch : (i + 1) * n_batch][0] >= threshold:
            threshold_count += 1
    print(threshold_count),print()
    return threshold_count

def get_size_matrix(size, threshold=5, n_simulations=29, n_batch=50):
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

data = np.loadtxt("data/evolution.txt")
size_check = data[1:, [0,-1]]
size = data[1:,0]

bins = np.arange(0, 100, 4)
for i, realization in enumerate(chunked(size, 50)):
	if i%5==0:
		plt.hist(realization, edgecolor="black",bins=bins, label=f"Immunes: {i+1}")
		plt.axvline(x=48, color="black", linestyle="dashed", linewidth=3)
		plt.legend()
		plt.show()

for i, simulation in enumerate(chunked(size, 50)):
	print(f"{i}.",outbreak_detector(simulation))

data_matrix = get_size_matrix(size)
print(data_matrix)

x_imnues = np.arange(1, 30, 1)

for i in range(29):
	break
	plt.scatter([x_imnues[i] for j in range(50)], data_matrix[i,:])

mean_size = np.array([data_matrix[i,:].mean() for i in range(0, 29)])
std_size = np.array([data_matrix[i,:].std() for i in range(0, 29)])

plt.errorbar(
    x_imnues, mean_size, yerr=std_size, fmt="o", color="blue", label="Mean Size"
)
plt.plot(x_imnues, mean_size, linewidth=3, color="red")
plt.axhline(y=48, color="black", linestyle="dashed", label="Simulation with 0 immunes", linewidth=3)

plt.legend()
plt.show()

