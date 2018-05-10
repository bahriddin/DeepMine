import numpy as np
import tensorflow as tf

np.random.seed(1)
tf.set_random_seed(1)


class DeepQNetwork:
    def __init__(self, env):
        self.n_actions = env.action_space.n
        self.n_features = len(env.reset())
        self.lr = 0.0001
        self.gamma = 0.999
        self.epsilon_max = 0.9
        self.replace_target_iter = 500
        self.memory_size = 500000
        self.batch_size = 1000
        self.e_greedy_increment = 0.0001
        self.epsilon = 0 if self.e_greedy_increment is not None else self.epsilon_max
        self.output_graph = False

        # total learning step
        self.learn_step_counter = 0

        # Count memory
        self.memory_counter = 0

        # initialize zero memory [s, a, r, s_]
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))

        # consists of [target net, evaluation net]
        self._build_net()

        self.path = "./dqn"  # The path to save our model to.
        self.saver = tf.train.Saver()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        if self.output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        self.cost_his = []

    def _build_net(self):
        # ------------------ all inputs ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # Input state
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # Input Next state
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # Input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # Input Action

        w_initializer, b_initiliazer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)

        # ------------------ build evaluation net ------------------
        with tf.variable_scope('eval_net'):
            e1 = tf.layers.dense(self.s, 64, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initiliazer, name='e1')
            self.q_eval = tf.layers.dense(e1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initiliazer, name='q')

        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(self.s_, 64, tf.nn.relu, kernel_initializer=w_initializer,
                                 bias_initializer=b_initiliazer, name='t1')
            self.q_next = tf.layers.dense(t1, self.n_actions, kernel_initializer=w_initializer,
                                          bias_initializer=b_initiliazer, name='t2')

        with tf.variable_scope('q_target'):
            q_target = self.r + self.gamma * tf.reduce_max(self.q_next, axis=1, name='Qmax_s_')  # shape=(None,)
            self.q_target = tf.stop_gradient(q_target)

        with tf.variable_scope('q_eval'):
            a_indices = tf.stack([tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a], axis=1)
            self.q_eval_wrt_a = tf.gather_nd(params=self.q_eval, indices=a_indices)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval_wrt_a, name='TD_error'))

        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))

        # replace old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):
        # to have batch dimension when feed into tf placeholder
        observation = observation[np.newaxis, :]

        if np.random.uniform() < self.epsilon:
            # forward feed the observation and get q value for every actions
            actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
            action = np.argmax(actions_value)
        else:
            action = np.random.randint(0, self.n_actions)

        return action

    def learn(self):
        # check to replace target parameters
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.target_replace_op)
            # print('\ntarget_params_replaced\n')

        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        batch_memory = self.memory[sample_index, :]

        _, cost = self.sess.run(
            [self._train_op, self.loss],
            feed_dict={
                self.s: batch_memory[:, :self.n_features],
                self.a: batch_memory[:, self.n_features],
                self.r: batch_memory[:, self.n_features + 1],
                self.s_: batch_memory[:, -self.n_features:],
            })

        self.cost_his.append(cost)

        # increasing epsilon
        self.epsilon = self.epsilon + self.e_greedy_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1

    def save_model(self, i):
        self.saver.save(self.sess, self.path + '/model-' + str(i) + '.ckpt')
        print("Saved Model")
