import gym
from policy_gradient import PolicyGradient
import matplotlib.pyplot as plt
import numpy as np
import time
import os
env = gym.make('LunarLander-v2')
env = env.unwrapped

# Policy gradient has high variance, seed for reproducability
env.seed(1)

show = False
RENDER_ENV = False
EPISODES = 50000
rewards = []
RENDER_REWARD_MIN = 5000

if __name__ == "__main__":

    # Load checkpoint
    load_version = 0
    save_version = load_version + 1
    # load_path = "output/weights/LunarLander/{}/LunarLander-v2.ckpt".format(load_version)
    # save_path = "output/weights/LunarLander/{}/LunarLander-v2.ckpt".format(save_version)
    learning_rate=0.001
    load_path=None
    save_path = str(learning_rate)
    reward_path = save_path + '/rlist'
    # if not os.path.exists(save_path):
    #     os.chmod(save_path, 0o777)
    #     os.makedirs(save_path)

    PG = PolicyGradient(
        n_x = env.observation_space.shape[0],
        n_y = env.action_space.n,
        learning_rate=learning_rate,
        reward_decay=0.99,
        load_path=load_path,
        save_path=save_path
    )


    for episode in range(EPISODES):

        observation = env.reset()
        episode_reward = 0

        tic = time.clock()

        while True:
            if RENDER_ENV: env.render()

            # 1. Choose an action based on observation
            action = PG.choose_action(observation)

            # 2. Take action in the environment
            observation_, reward, done, info = env.step(action)

            # 4. Store transition for training
            PG.store_transition(observation, action, reward)

            toc = time.clock()
            elapsed_sec = toc - tic
            if elapsed_sec > 20:
                done = True

            episode_rewards_sum = sum(PG.episode_rewards)
            if episode_rewards_sum < -250:
                done = True

            if done:
                episode_rewards_sum = sum(PG.episode_rewards)
                rewards.append(episode_rewards_sum)
                max_reward_so_far = np.amax(rewards)

                # print("==========================================")
                print("Episode: ", episode,"Seconds: ", elapsed_sec,"Reward: ", episode_rewards_sum)
                # print("Seconds: ", elapsed_sec)
                # print("Reward: ", episode_rewards_sum)
                # print("Max reward so far: ", max_reward_so_far)

                # 5. Train neural network
                discounted_episode_rewards_norm = PG.learn()

                # Periodically save the model.
                if episode % 1000 == 0 and PG.save_path is not None:
                    PG.save_model(episode)

                # Periodically save the model.
                if episode > 0 and episode % 200 == 0:
                    plt.close('all')
                    # rewards=np.load(rewards_file)
                    rMat = np.resize(np.array(rewards), [len(rewards) // 100, 100])
                    rMean = np.average(rMat, 1)
                    plt.plot(rMean)
                    plt.xlabel('Training episodes (hundreds)')
                    plt.ylabel('Average rewards every 100 episodes')
                    # plt.xticks(np.array([x for x in range(len(rMean))]))
                    if show:
                        plt.show(block=False)

                    plt.savefig(save_path + "/training.png")
                    np.save(reward_path, np.array(rewards))


                if max_reward_so_far > RENDER_REWARD_MIN: RENDER_ENV = True


                break

            # Save new observation
            observation = observation_
