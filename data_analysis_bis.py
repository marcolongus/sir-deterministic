import numpy as np
import matplotlib.pyplot as plt
from more_itertools import chunked
import itertools

n_simulations = 200
n_immunes = 20 # Numbers of immunes realizations

def outbreak_detector(simulation: np.array, n_batch=50):
	# simulation: batch of system evolution with the same seed
	# Store epidemic size at each entry. 
	outbreak = []
	outbreak_flag = False
	for i in range(n_batch-1):
		#print(i, simulation)
		if simulation[i] < simulation[i + 1]:
			outbreak.append(simulation[i + 1] - simulation[i])
			outbreak_flag = True
	return [outbreak, outbreak_flag]

# If epidemy is greatar than some initial value we consider it. 
def get_threshold(size: np.array, thres=5, n_sim=29, n_real=50):
    threshold_count = 0 
    for i in range(n_simulations):
        if size[i * n_real : (i + 1) * n_real][0] >= thres:
            threshold_count += 1

    print("Threshold_count: ",threshold_count),print()
    return threshold_count

# Get a matrix with variation of each epidemic
def get_size_matrix(size, threshold=1, n_simulations=29, n_batch=50):
	matrix_rows = get_threshold(size, thres=threshold, n_sim=n_simulations, n_real=n_batch)
	size_matrix = np.zeros(shape=(matrix_rows, n_batch))
	threshold_count = 0
	for i in range(n_simulations):
		simulation_i = size[i * n_batch : (i + 1) * n_batch]
		#print(simulation_i, np.array(simulation_i).std()), print()
		if simulation_i[0] >= threshold:
			size_matrix[threshold_count,:] = simulation_i
			threshold_count+=1
	return size_matrix

#==================================================================================#
# Main
#==================================================================================#
data = np.loadtxt("data/sim_batch_pl/evolution.txt")
start_point = data[0]
size = data[:,0] 


#=============================================#
# Plot histogram for targets simulations
#=============================================#
# For each n_simul = 200 simulations.
# There are n_realization = 20 with 0, 0, 1, ..., 19 immunes.

bins = np.arange(0, 100, 4)
for i, realization in enumerate(chunked(size, 20)):
	break
	if i%2==0:
		plt.hist(realization, edgecolor="black",bins=bins, label=f"Immunes: {i+1}, ")
		plt.axvline(x=np.array(realization).mean(), color="black", linestyle="dashed", linewidth=3)
		plt.legend()
		plt.show()


#==============================================#
# Outrbeaks
#==============================================#
outbreaks_counter = []
outbreaks_size = []
for i, simulation in enumerate(chunked(size, 20)):
	out_size, out_bool = outbreak_detector(simulation, n_immunes)
	outbreaks_counter.append(out_bool)
	outbreaks_size.append(out_size)

prob_outbreak = np.array(outbreaks_counter, dtype=int).sum() / n_simulations
print("Prob. Of Outbreak:", prob_outbreak)
flattened_size = list(itertools.chain.from_iterable(outbreaks_size))

bins = np.arange(0, 100, 4)
plt.hist(flattened_size, edgecolor="black", bins=bins)
plt.tight_layout()
plt.show()

#==================================================#
# Noise plot calculation
#==================================================#

data_matrix = get_size_matrix(size, n_simulations=200, n_batch=20)

print("DATA MATRIX")
print(data_matrix[0,1:].size)

x_imnues = np.arange(0, 19)

plt.xlim(0,18)
plt.ylim(0, 100)
plt.xlabel("immunes", fontsize=20)
plt.ylabel("epidemic size", fontsize=20)
for j in range(200):
	if outbreaks_counter[j] and np.array(outbreaks_size[j]).max()>52:
		plt.plot(x_imnues, data_matrix[j,1:], linewidth=3)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.tight_layout()
plt.savefig("video/pl_realizations_indep.png", dpi=300)
plt.show()



#===========================================================================#
# x_imnues = np.arange(1, 30, 1)

# mean_list = [data_matrix[i,:].mean() for i in range(0, 29)]
# std_list = [data_matrix[i,:].std() for i in range(0, 29)]

# mean_size = np.array(mean_list)
# std_size = np.array(std_list)

# plt.errorbar(
#     x_imnues, mean_size, yerr=std_size, fmt="o", color="blue", label="Mean Size"
# )

# mean_list.insert(0, start_point[0])
# x_imnues = np.arange(0, 30, 1)

# plt.xlabel("immunes", fontsize=25)
# plt.ylabel("epidemic size", fontsize=25)

# plt.xticks(fontsize=15)
# plt.yticks(fontsize=15)


# plt.plot(x_imnues, mean_list, linewidth=3, color="red")
# plt.axhline(y=start_point[0], color="black", linestyle="dashed", label="Simulation with 0 immunes", linewidth=3)
# plt.scatter(0, start_point[0], color='blue')
# plt.legend(fontsize=12)
# plt.tight_layout()
# plt.savefig("video/exp_noisy.png", dpi=300)
# plt.show()

