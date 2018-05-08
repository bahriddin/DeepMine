from __future__ import division
import gym
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce
from operator import mul
from DQN import DeepQNetwork

# env = gym.make('CartPole-v0')
env = gym.make('LunarLander-v2')
env.reset()
deepQNetwork = DeepQNetwork(env)
show = False # Whether to show the game or not
reward_path = 'rlist'

def processState(states):
    return np.reshape(states, [reduce(mul, list(env.observation_space.shape), 1)])

max_epLength = 1000  # The max allowed length of our episode.

# create lists to contain total rewards and steps per episode
jList = []
rList = []

step = 0
num_episodes = 50001
for i in range(num_episodes):
    # Reset environment and get first new observation
    if i > 0 and i % 5000 == 0:
        show = True
    else:
        show = False

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

        a = deepQNetwork.choose_action(s)

        s1, r, d, info = env.step(a)
        s1 = processState(s1)

        deepQNetwork.store_transition(s, a, r, s1)

        if step > 1000 and step % 10:
            deepQNetwork.learn()

        rAll += r
        s = s1

        if d == True:
            break
        step += 1


    jList.append(j)
    rList.append(rAll)

    # Periodically save the model.
    if i > 0 and i % 1000 == 0:
        deepQNetwork.save_model(i)

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

        np.save(reward_path, np.array(rList))

    if i % 10 == 0:
        print(deepQNetwork.learn_step_counter, np.mean(rList[-10:]), deepQNetwork.epsilon, deepQNetwork.memory_counter)