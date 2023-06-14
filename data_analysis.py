import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("data/sim_batch_uniform/evolution.txt")


size = data[:, 0]

inactive = np.array([i for i in range(0, 20)])
inactive[0] = 1
inactive = inactive - 1

print(inactive)
print()
print(100*"-")

plt.xlim(0,18)
plt.ylim(0, 80)

size_matrix = np.zeros(shape=(102, 20))
print(size_matrix)

plt.xlabel("# imnunes")
plt.ylabel("Size")

threshold_count = 0
for i in range(200):
	print("simulation:", i)
	size_simulation = size[i*20:(i+1)*20]
	print(size_simulation)
	print(100 * "-")
	if size_simulation[0] >= 5:
		size_matrix[threshold_count,:] = size_simulation
		threshold_count+=1
		plt.plot(inactive, size_simulation)

plt.show()
print(threshold_count)
print(size_matrix)

mean_size = [size_matrix[:,i].mean() for i in range(20)]
std_size = [size_matrix[:,i].std() for i in range(20)]

plt.xlim(0,18)
plt.ylim(0, 60)
plt.errorbar(inactive, mean_size, yerr=std_size, fmt='o', color='blue', label='Mean Size')
plt.plot(inactive, mean_size)
plt.legend()
plt.show()