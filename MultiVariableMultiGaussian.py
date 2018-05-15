import numpy as np
from scipy.stats import multivariate_normal as mn
from scipy.stats import rv_discrete
import matplotlib.pyplot as plt
import itertools

SAMPLE_SIZE = 20
NUM_CLUSTERS = 10
RANGE = (-10,10)

def init_data():
	t_means = (RANGE[1] - RANGE[0]) * np.random.rand(NUM_CLUSTERS, 2) - (RANGE[1] - RANGE[0]) / 2
	data = np.array([mn.rvs(t_means[i], np.eye(2), SAMPLE_SIZE) for i in range(NUM_CLUSTERS)])
	data = np.array(list(itertools.chain.from_iterable(data)))
	return t_means, data

def construct_and_sample(pmf_not_normed, sample_num=1):
	pmf = pmf_not_normed/sum(pmf_not_normed)
	rv = rv_discrete(values=(np.arange(0, len(pmf)),pmf))
	return rv.rvs(sample_num)

# Implements kmeans++ initialization algorithm
def init_means(samples):
	in_means = [samples[np.random.randint(0,len(samples)-1),:]]
	for i in range(1,NUM_CLUSTERS):
		cur_d = (np.linalg.norm(in_means[i-1] - samples, axis=1))**2
		in_means.append(samples[construct_and_sample(cur_d)])
	return np.array(in_means)



def hard_expect(cur_means, samples):
	s = np.tile(samples, (1, NUM_CLUSTERS))
	# print(s)
	c = cur_means.flatten()
	d = np.linalg.norm(np.abs(c - s).reshape(SAMPLE_SIZE*NUM_CLUSTERS, NUM_CLUSTERS, 2), axis=2)
	indeces = np.argmin(d, axis=1)
	# print([samples[indeces == i,:] for i in range(NUM_CLUSTERS)])
	return [samples[indeces == i,:] for i in range(NUM_CLUSTERS)]


def maximize(weighted, cur_means):
	ret = list()
	for i in range(len(weighted)):
		# if len(weighted[i]) != 0:
		ret.append(np.mean(weighted[i], axis=0))
		# else:
		# 	ret.append(cur_means[i])
	print(np.array(ret))
	return np.array(ret)



def run(samples, true_means):
	cur_means = init_means(samples)

	plt.scatter(samples[:, 0], samples[:, 1], color="red")
	plt.scatter(true_means[:,0], true_means[:,1], color="yellow")
	plt.scatter(cur_means[:,0], cur_means[:,1], color="blue")

	plt.show()

	old_means = None
	cur_weighted = None
	c = 0
	while old_means is None or not np.array_equal(old_means, cur_means):
		old_means = cur_means
		cur_weighted = hard_expect(cur_means, samples)
		cur_means = maximize(cur_weighted, cur_means)
		if c % 1 == 0:
			plt.cla()
			plt.scatter(samples[:,0], samples[:,1], color="red")
			plt.scatter(cur_means[:,0], cur_means[:,1], color="blue")
			plt.scatter(true_means[:,0], true_means[:,1], color="yellow")
			plt.show()
		c += 1
	plt.cla()
	plt.scatter(samples[:, 0], samples[:, 1], color="red")
	plt.scatter(cur_means[:, 0], cur_means[:, 1], color="blue")
	plt.scatter(true_means[:, 0], true_means[:, 1], color="yellow")
	plt.show()
	print("Iterations:", c)
	print("Cur means", cur_means)


true_means, samples = init_data()
run(samples, true_means)