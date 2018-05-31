import numpy as np
import matplotlib as mpl
mpl.use("TkAgg")  # Use TKAgg to show figures
import matplotlib.pyplot as plt
# rewards=np.load("losslist.npy")
# rewards=np.load("steplist.npy")
rewards=np.load("rlist.npy")

rMat = np.resize(np.array(rewards), [len(rewards) // 100, 100])
rMean = np.average(rMat, 1)
plt.plot(rMean)
plt.xlabel('Training episodes (hundreds)')
plt.ylabel('Average rewards every 100 episodes')
# plt.xticks(np.array([x for x in range(len(rMean))]))
# if RENDER:
plt.show()