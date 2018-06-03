import numpy as np

arr = np.load('dqn-per-final2/experiment_results.npy')
arr = np.sort(arr)
print(arr)