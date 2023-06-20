import numpy as np
import matplotlib.pyplot as plt

<<<<<<< HEAD
def outbreak_detector(simulation):
	outbreak = []
	outbreak_flag = False
	for i in range(19):
		if simulation[i]<simulation[i+1]:
			outbreak.append(simulation[i+1]-simulation[i])
			outbreak_flag = True
	return [outbreak, outbreak_flag]

#=======================================================#
# Data definition
#=======================================================#
data = np.loadtxt("data/sim_batch_exp/evolution.txt")

size = data[:, 0]
inactive = np.array([i for i in range(0, 20)])
inactive[0] = 1
inactive = inactive - 1
=======
>>>>>>> ac07cd8dfb3bd50ae09eadb7c07e8bc54869c9f4

def outbreak_detector(simulation):
    outbreak = []
    outbreak_flag = False
    for i in range(19):
        if simulation[i] < simulation[i + 1]:
            outbreak.append(simulation[i + 1] - simulation[i])
            outbreak_flag = True
    return [outbreak, outbreak_flag]

def get_matrix_size(size: np.array):
threshold_count = 0
for i in range(200):
    if size[i * 20 : (i + 1) * 20][0] >= 5:
        threshold_count += 1

<<<<<<< HEAD
print("threshold_count:", threshold_count )
=======
# =======================================================#
# Data definition
# =======================================================#
data = np.loadtxt("data/sim_batch_uniform/evolution.txt")
size = data[:, 0]
threshold_count = get_matrix_size(size)
>>>>>>> ac07cd8dfb3bd50ae09eadb7c07e8bc54869c9f4
size_matrix = np.zeros(shape=(threshold_count, 20))
print(100 * "-")

# =======================================================#
# Analysis
# =======================================================#
outbreaks = []
outbreaks_histogram = []
threshold_count = 0

inactive = np.array([i for i in range(0, 20)])
inactive[0] = 1
inactive = inactive - 1

for i in range(200):
    print("simulation:", i)
    size_simulation = size[i * 20 : (i + 1) * 20]
    result = outbreak_detector(size_simulation)
    print(size_simulation)
    print(result)
    outbreaks.append(result[1])
    outbreaks_histogram.append(result[0])
    print(100 * "-")

    # Save simulations where epidemic has ocurred
    if size_simulation[0] >= 5:
        size_matrix[threshold_count, :] = size_simulation
        threshold_count += 1

    # Plot simulations with oubreaks
    if size_simulation[0] >= 5:
        if result[1] and max(result[0]) > 52:
            plt.plot(inactive, size_simulation, label=i)
            plt.scatter(inactive, size_simulation, label=i)


plt.xlabel("# imnunes")
plt.ylabel("Size")
plt.ylim(0, 100)
plt.xlim(0, 18)
plt.show()


print("\nPercentege of outbreak due to immunization:")
print(np.array(outbreaks, dtype=int).sum() / 200)

# =======================================================#
# Statistics
# =======================================================#
print("\nthreshold_count and matrix:", threshold_count)

mean_size = [size_matrix[:, i].mean() for i in range(20)]
std_size = [size_matrix[:, i].std() for i in range(20)]

# =======================================================#
# Plot histogram
# =======================================================#
plt.xlim(0, 18)
plt.ylim(0, 100)
plt.errorbar(
    inactive, mean_size, yerr=std_size, fmt="o", color="blue", label="Mean Size"
)
plt.plot(inactive, mean_size)
plt.legend()
plt.show()


#=======================================================#
# Histogram
#=======================================================#
to_histogram = []
for element in outbreaks_histogram:
    for outbreak_element in element:
        to_histogram.append(outbreak_element)

bins = [i for i in range(80)]
histo = np.histogram(to_histogram, bins=bins)
bins_result = histo[1]

# print(bins_result)        
# print(histo)

plt.hist(to_histogram, bins=bins_result, alpha=0.7, edgecolor="black")
        
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

<<<<<<< HEAD
plt.tight_layout()
plt.show()
=======
plt.hist(to_histogram, edgecolor='black', alpha=0.7)
plt.show()
>>>>>>> ac07cd8dfb3bd50ae09eadb7c07e8bc54869c9f4
