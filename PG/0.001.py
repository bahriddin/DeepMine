import gym
from policy_gradient import PolicyGradient

import numpy as np
import time

env = gym.make('LunarLander-v2')
env = env.unwrapped


# 训练了20000 episode, 数据存在12, 没开env


# Policy gradient has high variance, seed for reproducability
env.seed(1)

print("env.action_space", env.action_space)
print("env.observation_space", env.observation_space)
print("env.observation_space.high", env.observation_space.high)
print("env.observation_space.low", env.observation_space.low)


RENDER_ENV = False
EPISODES = 50000
# max_eplength=1000
rewards = []
RENDER_REWARD_MIN = 5000

if __name__ == "__main__":

    # Load checkpoint
    load_version = 12
    save_version = load_version + 1
    load_path = "output/weights/LunarLander/{}/LunarLander-v2.ckpt".format(load_version)
    # load_path=None
    save_path = "output/weights/LunarLander/{}/LunarLander-v2.ckpt".format(save_version)
    rewards_path="rewards_lr_001"
    rewards_file="rewards_lr_001.npy"

    PG = PolicyGradient(
        n_x = env.observation_space.shape[0],
        n_y = env.action_space.n,
        learning_rate=0.001,
        reward_decay=0.99,
        load_path=load_path,
        save_path=save_path
    )

    # create lists to contain total rewards and steps per episode
    jList = []
    rList = []


    for episode in range(EPISODES):

        observation = env.reset()
        episode_reward = 0

        tic = time.clock()
        j=0

        while True:
        # while i<max_eplength:
            j+=1
            if RENDER_ENV: env.render()

            # 1. Choose an action based on observation
            action = PG.choose_action(observation)

            # 2. Take action in the environment
            observation_, reward, done, info = env.step(action)

            # 4. Store transition for training
            PG.store_transition(observation, action, reward)

            toc = time.clock()
            elapsed_sec = toc - tic
            if elapsed_sec > 120:
                done = True

            # if j>max_eplength:
            #     done=True

            episode_rewards_sum = sum(PG.episode_rewards)
            if episode_rewards_sum < -250:
                done = True

            if done:
                episode_rewards_sum = sum(PG.episode_rewards)

                rewards.append(episode_rewards_sum)
                jList.append(j)
                max_reward_so_far = np.amax(rewards)

                print("==========================================")
                print("Episode: ", episode)
                print("Seconds: ", elapsed_sec)
                print("Reward: ", episode_rewards_sum)
                print("Max reward so far: ", max_reward_so_far)

                # 5. Train neural network
                discounted_episode_rewards_norm = PG.learn()

                if max_reward_so_far > RENDER_REWARD_MIN: RENDER_ENV = True


                break

            # Save new observation
            observation = observation_

        # Periodically save the model.
        if episode > 0 and episode % 200== 0:
            # Save checkpoint
            if PG.save_path is not None:
                save_path = PG.saver.save(PG.sess, PG.save_path)
                print("Model saved in file: %s" % save_path)

            # RL.save_model(i)
            import matplotlib as mpl

            mpl.use("TkAgg")  # Use TKAgg to show figures
            import matplotlib.pyplot as plt

            plt.close('all')
            np.save(rewards_path,np.array(rewards))
            # rewards=np.load(rewards_file)
            rMat = np.resize(np.array(rewards), [len(rewards) // 100, 100])
            rMean = np.average(rMat, 1)
            plt.plot(rMean)
            plt.xlabel('Training episodes (hundreds)')
            plt.ylabel('Average rewards every 100 episodes')
            # plt.xticks(np.array([x for x in range(len(rMean))]))
            # if RENDER:
            plt.show(block=False)
            plt.savefig("plt001.png")
