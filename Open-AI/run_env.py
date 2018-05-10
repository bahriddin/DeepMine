from __future__ import division
import gym
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from operator import mul
from Double_DQN import RL

# env = gym.make('CartPole-v0')
env = gym.make('LunarLander-v2')
env.reset()
RL = RL(env)
show = True # Whether to show the game or not
reward_path = 'rlist'

def processState(states):
    return np.reshape(states, [reduce(mul, list(env.observation_space.shape), 1)])

max_epLength = 1000  # The max allowed length of our episode.

# create lists to contain total rewards and steps per episode
jList = []
rList = []


i = 0
while True:
    # Reset environment and get first new observation
    s = env.reset()
    s = processState(s)
    d = False
    rAll = 0
    j = 0
    # The Q-Network
    while j < max_epLength:  # If the agent takes longer than 1000 moves to reach the end, end the trial.
        if show:
            env.render()
        j += 1

        a = RL.choose_action(s)

        s1, r, d, info = env.step(a)
        s1 = processState(s1)

        RL.learn(s, a, r, s1, d)

        rAll += r
        s = s1

        if d == True:
            break


    jList.append(j)
    rList.append(rAll)

    # Periodically save the model.
    if i > 0 and i % 1000 == 0:
        RL.save_model(i)

    if i > 0 and i % 100 == 0:
        plt.close('all')
        rMat = np.resize(np.array(rList), [len(rList) // 100, 100])
        rMean = np.average(rMat, 1)
        plt.plot(rMean)
        plt.xlabel('Training episodes (hundreds)')
        plt.ylabel('Average rewards every 100 episodes')
        # plt.xticks(np.array([x for x in range(len(rMean))]))
        if show:
            plt.show(block=False)
        plt.savefig("training.png")

        np.save(reward_path,np.array(rList))

    if i % 1 == 0:
        print(RL.total_steps, np.mean(rList[-1:]), RL.e)
    i += 1