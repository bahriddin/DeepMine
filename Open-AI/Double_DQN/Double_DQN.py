from __future__ import division
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import scipy.misc
import os
from functools import reduce
from operator import mul


class RL():
    def __init__(self, env, learning_rate = 0.0001,e=0.1,path='',load_model=False):
        self.batch_size = 512  # How many experiences to use for each training step.
        self.update_freq = 10  # How often to perform a training step.
        self.learning_rate = learning_rate
        self.y = .99  # Discount factor on the target Q-values
        self.startE = 1  # Starting chance of random action
        self.endE = e  # Final chance of random action
        self.annealing_steps = 10000.  # How many steps of training to reduce startE to endE.
        self.num_episodes = 10000  # How many episodes of game environment to train network with.
        self.pre_train_steps = 10000  # How many steps of random actions before training begins.
        self.total_steps = 0

        self.flatten_shape = reduce(mul, list(env.observation_space.shape), 1)
        self.h_size = self.flatten_shape  # 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
        self.n_actions = env.action_space.n

        tf.reset_default_graph()
        self.mainQN = self.Qnetwork(self.h_size,self.n_actions,self.learning_rate )
        self.targetQN = self.Qnetwork(self.h_size,self.n_actions,self.learning_rate )

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

        self.path = path  # The path to save our model to.
        self.loss_history = []

        # Make a path for our model to be saved in.
        if not os.path.exists(self.path):
            os.makedirs(self.path)

        self.saver = tf.train.Saver(max_to_keep = None)
        self.save_count = 0


        tau = 0.001  # Rate to update target network toward primary network
        self.trainables = tf.trainable_variables()
        self.targetOps = self.updateTargetGraph(self.trainables, tau)


        self.episodeBuffer = self.experience_buffer()
        self.myBuffer = self.experience_buffer()

        # Set the rate of random action decrease.
        self.e = self.startE
        self.stepDrop = (self.startE - self.endE) / self.annealing_steps

        # load_model = False  # Whether to load a saved model.
        if load_model == True:
            print('Loading Model...')
            ckpt = tf.train.get_checkpoint_state(self.path)
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            self.e = 0
            self.pre_train_steps = 0

    class Qnetwork():
        def __init__(self, h_size, n_actions, learning_rate):
            self.n_actions = n_actions
            self.learning_rate = learning_rate
            # The network recieves a frame from the game, flattened into an array.
            # It then resizes it and processes it through four convolutional layers.
            self.scalarInput = tf.placeholder(shape=[None, h_size ], dtype=tf.float32)
            w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
            self.l1 = tf.layers.dense(self.scalarInput, 50, tf.nn.relu, kernel_initializer=w_initializer, bias_initializer=b_initializer)
            xavier_init = tf.contrib.layers.xavier_initializer()
            self.Qout = tf.layers.dense(self.l1, self.n_actions, kernel_initializer=xavier_init,
                                          bias_initializer=b_initializer)

            # self.streamA = slim.flatten(self.l1)  # streamAC)
            # self.streamV = slim.flatten(self.l1)  # streamVC)
            # xavier_init = tf.contrib.layers.xavier_initializer()
            # self.AW = tf.Variable(xavier_init([50 , self.n_actions]))
            # self.VW = tf.Variable(xavier_init([50 , 1]))
            # self.Advantage = tf.matmul(self.streamA, self.AW)
            # self.Value = tf.matmul(self.streamV, self.VW)

            # Then combine them together to get our final Q-values.
            # self.Qout = self.Value + tf.subtract(self.Advantage,
            #                                      tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
            self.predict = tf.argmax(self.Qout, 1)

            # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
            self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
            self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
            self.actions_onehot = tf.one_hot(self.actions, self.n_actions, dtype=tf.float32)

            self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

            self.td_error = tf.square(self.targetQ - self.Q)
            self.loss = tf.reduce_mean(self.td_error)
            self.trainer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
            self.updateModel = self.trainer.minimize(self.loss)

    def choose_action(self, s):
        # Choose an action by greedily (with e chance of random action) from the Q-network
        if np.random.rand(1) < self.e or self.total_steps < self.pre_train_steps:
            a = np.random.randint(0, self.n_actions)
        else:
            a = self.sess.run(self.mainQN.predict, feed_dict={self.mainQN.scalarInput: [s]})[0]
        a = np.int64(a)
        self.total_steps += 1
        return a

    def learn(self, s, a, r, s1, d):

        self.episodeBuffer.add(
            np.reshape(np.array([s, a, r, s1, d]), [1, 5]))  # Save the experience to our episode buffer.

        if self.total_steps > self.pre_train_steps:
            if self.e > self.endE:
                self.e -= self.stepDrop
            if self.total_steps % (self.update_freq) == 0:
                trainBatch = self.myBuffer.sample(self.batch_size)  # Get a random batch of experiences.
                # Below we perform the Double-DQN update to the target Q-values
                Q1 = self.sess.run(self.mainQN.predict, feed_dict={self.mainQN.scalarInput: np.vstack(trainBatch[:, 3])})
                Q2 = self.sess.run(self.targetQN.Qout, feed_dict={self.targetQN.scalarInput: np.vstack(trainBatch[:, 3])})
                end_multiplier = -(trainBatch[:, 4] - 1)
                doubleQ = Q2[range(self.batch_size), Q1]
                targetQ = trainBatch[:, 2] + (self.y * doubleQ * end_multiplier)
                # Update the network with our target values.
                _ ,loss = self.sess.run([self.mainQN.updateModel,self.mainQN.loss], \
                             feed_dict={self.mainQN.scalarInput: np.vstack(trainBatch[:, 0]), self.mainQN.targetQ: targetQ,
                                        self.mainQN.actions: trainBatch[:, 1]})
                self.loss_history.append(loss)
                self.updateTarget(self.targetOps)  # Update the target network toward the primary network.


        if d == True:
            self.myBuffer.add(self.episodeBuffer.buffer)

    def get_loss_history(self):
        return np.array(self.loss_history)

    def updateTargetGraph(self, tfVars, tau):
        total_vars = len(tfVars)
        op_holder = []
        for idx, var in enumerate(tfVars[0:total_vars // 2]):
            op_holder.append(tfVars[idx + total_vars // 2].assign(
                (var.value() * tau) + ((1 - tau) * tfVars[idx + total_vars // 2].value())))
        return op_holder

    def updateTarget(self, op_holder):
        for op in op_holder:
            self.sess.run(op)

    def save_model(self,i):
        self.saver.save(self.sess, self.path +  '/model-' + str(i) + '.ckpt')
        # if i % 5000 == 0:
        #     self.save_count += 1
        #     self.saver = tf.train.Saver()
        print("Saved Model")

    class experience_buffer():
        def __init__(self, buffer_size=500000):
            self.buffer = []
            self.buffer_size = buffer_size

        def add(self, experience):
            if len(self.buffer) + len(experience) >= self.buffer_size:
                # offset = np.random.randint(0, int(len(self.buffer)/len(experience)))
                # self.buffer = self.buffer[0:(offset-1)*len(experience)]+self.buffer[offset*len(experience):]
                self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
            self.buffer.extend(experience)

        def sample(self, size):
            return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])