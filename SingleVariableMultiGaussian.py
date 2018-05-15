import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt


SAMPLE_SIZE = 50
NUM_CLUSTERS = 3
RANGE = (-10,10)
SD = 1



def hard_expect(cur_means, samples):
	# cur_means = cur_means.reshape((NUM_CLUSTERS, 1))
	# print(samples.shape)
	s = np.tile(samples, (NUM_CLUSTERS, 1))
	# print(np.abs(cur_means - s))
	indeces = np.argmin(np.abs(cur_means - s), axis=0)
	# print(indeces)
	# print([samples[:,indeces == i] for i in range(NUM_CLUSTERS)])
	return [samples[:,indeces == i] for i in range(NUM_CLUSTERS)]


def maximize(weighted):
	ret = list()
	# print(weighted)
	for w in weighted:
		ret.append([np.mean(w)])
	return np.array(ret)


def run(samples, true_means):
	cur_means = (RANGE[1] - RANGE[0]) * np.random.rand(NUM_CLUSTERS,1) - (RANGE[1] - RANGE[0]) / 2
	old_means = None
	cur_weighted = None
	c = 0
	while old_means is None or not np.array_equal(old_means, cur_means):
		old_means = cur_means
		cur_weighted = hard_expect(cur_means, samples)
		cur_means = maximize(cur_weighted)
		if c % 10 == 0:
			plt.cla()
			plt.scatter(samples, np.zeros_like(samples), color="red")
			plt.scatter(cur_means.tolist(), np.zeros_like(cur_means.tolist()), color="blue")
			plt.scatter(true_means.tolist(), np.zeros_like(true_means.tolist()), color="yellow")
			plt.show()
		c += 1
	plt.cla()
	plt.scatter(samples, np.zeros_like(samples), color="red")
	plt.scatter(cur_means.tolist(), np.zeros_like(cur_means.tolist()), color="blue")
	plt.scatter(true_means.tolist(), np.zeros_like(true_means.tolist()), color="yellow")
	plt.show()
	print("Iterations:", c)
	print("Cur means", cur_means)



true_means = (RANGE[1] - RANGE[0])*np.random.rand(NUM_CLUSTERS, 1) - (RANGE[1] - RANGE[0])/2
print("True means", true_means)
samples = np.array([np.random.normal(true_means[i//SAMPLE_SIZE], SD) for i in range(SAMPLE_SIZE*NUM_CLUSTERS)]).reshape((1,SAMPLE_SIZE*NUM_CLUSTERS))
# np.random.shuffle(samples)
run(samples, true_means)






