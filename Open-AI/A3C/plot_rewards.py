import numpy as np
import matplotlib as mpl
mpl.use("TkAgg")  # Use TKAgg to show figures
import matplotlib.pyplot as plt

def plot_reward(rewards_file):

    rewards=np.load(rewards_file)
    # print(rewards)
    rMat = np.resize(np.array(rewards), [len(rewards) // 100, 100])
    rMean = np.average(rMat, 1)
    plt.plot(rMean)
    plt.xlabel('Training episodes (hundreds)')
    plt.ylabel('Average rewards every 100 episodes')
    # plt.xticks(np.array([x for x in range(len(rMean))]))
    # if RENDER:
    plt.show()
    # plt.savefig("plt_a0005_c001_softmax.png")

# plot_reward("rewards_a0005_c001.npy")
# plot_reward("raw_A3C_rlist.npy")
plot_reward("raw_A3C_slist.npy")