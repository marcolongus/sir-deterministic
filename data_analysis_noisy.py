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

# If epidemy is greatar than some initial value we consider it. 
def get_threshold(size: np.array, threshold=1, n_simulations=29, n_batch=50):
    threshold_count = 0 
    for i in range(n_simulations):
        if size[i * n_batch : (i + 1) * n_batch][0] >= threshold:
            threshold_count += 1
    print("threshold_count: ",threshold_count),print()
    return threshold_count

# Get a matrix with variation of each epidemic
def get_size_matrix(size, threshold=0, n_simulations=29, n_batch=50):
	size_matrix = np.zeros(shape=(get_threshold(size), n_batch))
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
data = np.loadtxt("data/exp_noisy_bis/evolution.txt")
start_point = data[0]
size_check = data[1:, [0,-1]]
size = data[1:,0] 
#print(size)

#=============================================#
# Plot histogram for targets simulations
#=============================================#
# For each n_simul = 30 simulations with n_simul immunes.
# There are n_realization = 50 realization where the n_simul immunes 
# where choosen at random at the start of the simulation.
bins = np.arange(0, 100, 4)
for i, realization in enumerate(chunked(size, 50)):
	plt.hist(realization, edgecolor="black",bins=bins, label=f"Immunes: {i+1}, ")
	plt.axvline(x=np.array(realization).mean(), color="black", linestyle="dashed", linewidth=3)
	plt.legend()
	plt.ylabel("epidemic size", fontsize=25)
	plt.savefig(f"video/histo_noise{i+1}.png", dpi=300)
	plt.show()

#=============================================#
# Plot histogram for all simulations
#=============================================#
# plt.hist(size, edgecolor="black",bins=bins, label=f"Immunes: {i+1}")
# plt.axvline(x=start_point[0], color="black", linestyle="dashed", linewidth=3)
# plt.legend()
# plt.show()

#=============================================#
# Oubreak detector
#=============================================#
for i, simulation in enumerate(chunked(size, 50)):
	break
	print(f"{i}.",outbreak_detector(simulation))

#==============================================#
# Noise plot calculation
#==============================================#
data_matrix = get_size_matrix(size)

print("DATA MATRIX")
print(data_matrix)


#==============================================#
# Noise plot calculation
#==============================================#

x_imnues = np.arange(1, 30, 1)

mean_list = [data_matrix[i,:].mean() for i in range(0, 29)]
std_list = [data_matrix[i,:].std() for i in range(0, 29)]

mean_size = np.array(mean_list)
std_size = np.array(std_list)

plt.errorbar(
    x_imnues, mean_size, yerr=std_size, fmt="o", color="blue", label="Mean Size"
)

mean_list.insert(0, start_point[0])
x_imnues = np.arange(0, 30, 1)

plt.xlabel("immunes", fontsize=25)
plt.ylabel("epidemic size", fontsize=25)

plt.xticks(fontsize=15)
plt.yticks(fontsize=15)


plt.plot(x_imnues, mean_list, linewidth=3, color="red")
plt.axhline(y=start_point[0], color="black", linestyle="dashed", label="Simulation with 0 immunes", linewidth=3)
plt.scatter(0, start_point[0], color='blue')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig("video/exp_noisy_bis.png", dpi=300)
plt.show()

