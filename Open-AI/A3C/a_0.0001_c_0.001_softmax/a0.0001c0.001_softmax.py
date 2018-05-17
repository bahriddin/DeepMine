"""
Asynchronous Advantage Actor Critic (A3C) with continuous action space, Reinforcement Learning.

The Pendulum example. Convergence promised, but difficult environment, this code hardly converge.

View more on [莫烦Python] : https://morvanzhou.github.io/tutorials/

Using:
tensorflow 1.0
gym 0.8.0
"""

import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt
import time

# action=0,  cell_state not nan, still change, but prob_weights= [[ nan nan nan nan]], happens on 2000-4000.
# after that, prob_weights will always be nan. prob_weights=np.nan_to_num(prob_weights) can not slove problems
# It will get ValueError: probabilities do not sum to 1. But we can leave it as an assert. We should find the reason why it will become nan.
# with print(prob_weights), this error occurs on Ep: 1193

start=time.clock()

GAME = 'LunarLander-v2'
OUTPUT_GRAPH = True
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()
MAX_GLOBAL_EP = 50000
GLOBAL_NET_SCOPE = 'Global_Net'
UPDATE_GLOBAL_ITER = 5
GAMMA = 0.99
ENTROPY_BETA = 0.001   # not useful in this case
LR_A = 0.0001   # learning rate for actor
LR_C = 0.001    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0

# Load checkpoint
# load_version = 1
# save_version = load_version + 1
# rewards_path="rewards_a0005_c001"
# # load_path = "{}/AC3-v2.ckpt".format(load_version)
# load_path=None
# save_path = "{}/AC3-v2.ckpt".format(save_version)

load_path=None
save_path = 'a'+str(LR_A)+'c'+str(LR_C)
reward_path = save_path + '/rlist'

env = gym.make(GAME)

N_S = env.observation_space.shape[0]
N_A = env.action_space.n
del env

# 这个 class 可以被调用生成一个 global net.
# 也能被调用生成一个 worker 的 net, 因为他们的结构是一样的,
# 所以这个 class 可以被重复利用.
class ACNet(object):
    def __init__(self, scope, SESS,globalAC=None):
        # 当创建 worker 网络的时候, 我们传入之前创建的 globalAC 给这个 worker
        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S') # [None, N_S] coloumn is N_S (observation_space), row to be determined
                self._build_net(N_A) # N_A: action_space
                self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')

                # 'Saver' op to save and restore all the variables
                self.saver = tf.train.Saver()
                # Restore model
                if load_path is not None:
                    self.saver.restore(SESS, load_path)
                    print("Model loaded")
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')  # [None, N_S] coloumn is N_S (observation_space), row to be determined
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                self.a_prob, self.v = self._build_net(N_A)

                # 接着计算 critic loss 和 actor loss
                # 用这两个 loss 计算要推送的 gradients

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.a_prob) * tf.one_hot(self.a_his, N_A, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v = log_prob * td
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob), axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)
                    # 因为我们想不断增加这个 exp_v (动作带来的额外价值),
                    # 所以我们用过 minimize(-exp_v) 的方式达到
                    # maximize(exp_v) 的目的

                with tf.name_scope('local_grad'):
                    self.a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
                    self.c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'): # 同步
                with tf.name_scope('pull'): # 从 global_net 中获取到最新的参数.
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'): # 将自己的个人更新推送去 global_net.
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, n_a):
        # 在这里搭建 Actor 和 Critic 的网络
        w_init = tf.random_normal_initializer(mean=0., stddev=.01)
        with tf.variable_scope('critic'):
            cell_size = 64
            s = tf.expand_dims(self.s, axis=1,
                               name='timely_input')  # [time_step, feature] => [time_step, batch, feature]
            rnn_cell = tf.contrib.rnn.BasicRNNCell(cell_size) # state_size = 64
            self.init_state = rnn_cell.zero_state(batch_size=1, dtype=tf.float32) # 通过zero_state得到一个全0的初始状态，形状为(batch_size, state_size)
            # state size 是隐层的大小
            outputs, self.final_state = tf.nn.dynamic_rnn(
                cell=rnn_cell, inputs=s, initial_state=self.init_state, time_major=True)
            # time_major: The shape format of the inputs and outputs Tensors.
            # If true, these Tensors must be shaped [max_time, batch_size, depth].
            # If false, these Tensors must be shaped [batch_size, max_time, depth].
            # Using time_major = True is a bit more efficient because it avoids transposes at the beginning and end of the RNN calculation.
            # However, most TensorFlow data is batch-major, so by default this function accepts input and emits output in batch-major form.

            # in BasicRNNCell, output size=state size, so it needs to be transform to the real output
            cell_out = tf.reshape(outputs, [-1, cell_size], name='flatten_rnn_outputs')  # joined state representation
            # 其中shape为一个列表形式，特殊的一点是列表中可以存在-1。
            # -1代表的含义是不用我们自己指定这一维的大小，函数会自动计算，但列表中只能存在一个-1。
            # （当然如果存在多个-1，就是一个存在多解的方程了）
            l_c = tf.layers.dense(cell_out, 200, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        with tf.variable_scope('actor'):
            cell_out = tf.stop_gradient(cell_out, name='c_cell_out')
            # l_a = tf.layers.dense(cell_out, 300, tf.nn.relu6, kernel_initializer=w_init, name='la')
            l_a = tf.layers.dense(cell_out, 300, tf.nn.softmax, kernel_initializer=w_init, name='la')
            # 6 is just an arbitrary value chosen according to the number of bits
            # you want to be able to compress your network's trained parameters into.
            a_prob = tf.layers.dense(l_a, n_a, tf.nn.softmax, kernel_initializer=w_init, name='ap') # n_a : action space
        # return 均值，方差，state value
        return a_prob, v

    def update_global(self, feed_dict):  # run by a local # push
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local # pull
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s, cell_state):  # run by a local
        # print("self.s:",s[np.newaxis, :])
        # print("self.init_state:",cell_state)
        # print("self.a_prob:", self.a_prob)
        prob_weights, cell_state = SESS.run([self.a_prob, self.final_state], feed_dict={self.s: s[np.newaxis, :],
                                                                            self.init_state: cell_state})
        # print("prob_weights:",prob_weights)
        prob_weights=np.nan_to_num(prob_weights)
        # print(sum(prob_weights[0])-0.1)
        if sum(prob_weights[0])<0.1:
            print("Probabilities do not sum to 1")
        # print("cell_state:",cell_state)
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        # print("action:",action)
        return action, cell_state


class Worker(object):
    def __init__(self, name, SESS,globalAC):
        self.env = gym.make(GAME) # 创建自己的环境
        self.name = name # 自己的名字
        self.AC = ACNet(name,SESS, globalAC) # 自己的 local net, 并绑定上 globalAC

    def work(self):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        r_scale = 100
        buffer_s, buffer_a, buffer_r = [], [], []
        while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
            s = self.env.reset()
            ep_r = 0
            ep_t = 0
            rnn_state = SESS.run(self.AC.init_state)  # zero rnn state at beginning
            keep_state = rnn_state.copy()  # keep rnn state for updating global net
            while True:
                # if self.name == 'W_0' and total_step % 10 == 0:
                #     self.env.render()
                a, rnn_state_ = self.AC.choose_action(s, rnn_state)  # get the action and next rnn state
                s_, r, done, info = self.env.step(a)
                if r == -100: r = -10
                ep_r += r
                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append(r/r_scale)

                # 每 UPDATE_GLOBAL_ITER 步 或者回合完了, 进行 sync 操作
                if total_step % UPDATE_GLOBAL_ITER == 0 or done:   # update global and assign to local net
                    # 获得用于计算 TD error 的 下一 state 的 value
                    if done:
                        v_s_ = 0   # terminal
                    else:
                        v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :], self.AC.init_state: rnn_state_})[0,0]
                    buffer_v_target = [] # 下 state value 的缓存, 用于算 TD
                    for r in buffer_r[::-1]:    # reverse buffer r   # 进行 n_steps forward view
                        v_s_ = r + GAMMA * v_s_
                        buffer_v_target.append(v_s_)
                    buffer_v_target.reverse()

                    buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                    feed_dict = {
                        self.AC.s: buffer_s,
                        self.AC.a_his: buffer_a,
                        self.AC.v_target: buffer_v_target,
                        self.AC.init_state: keep_state,
                    }

                    self.AC.update_global(feed_dict)  # 推送更新去 globalAC

                    buffer_s, buffer_a, buffer_r = [], [], [] # 清空缓存
                    self.AC.pull_global()  # 获取 globalAC 的最新参数
                    keep_state = rnn_state_.copy()  # replace the keep_state as the new initial rnn state_

                s = s_
                total_step += 1
                rnn_state = rnn_state_  # renew rnn state
                ep_t += 1
                if done:
                    if len(GLOBAL_RUNNING_R) == 0:  # record running episode reward
                        GLOBAL_RUNNING_R.append(ep_r)
                    else:
                        GLOBAL_RUNNING_R.append(0.99 * GLOBAL_RUNNING_R[-1] + 0.01 * ep_r)
                    if not self.env.unwrapped.lander.awake: solve = '| Landed'
                    else: solve = '| ------'
                    print(
                        self.name,
                        "Ep:", GLOBAL_EP,
                        solve,
                        "| Ep_r: %i" % GLOBAL_RUNNING_R[-1],
                          )
                    GLOBAL_EP += 1 # 加一回合
                    break

if __name__ == "__main__":
    SESS = tf.Session()

    with tf.device("/cpu:0"):
        # OPT_A,OPT_C 在ACNet中用到, if scope != GLOBAL_NET_SCOPE:
        OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
        OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
        GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE,SESS)  # we only need its params  # 建立 Global AC
        workers = []
        # Create worker
        for i in range(N_WORKERS): # 创建 worker, 之后在并行
            i_name = 'W_%i' % i   # worker name
            workers.append(Worker(i_name, SESS,GLOBAL_AC)) # 每个 worker 都有共享这个 global AC

    COORD = tf.train.Coordinator() # Tensorflow 用于并行的工具
    SESS.run(tf.global_variables_initializer())

    if OUTPUT_GRAPH:
        if os.path.exists(LOG_DIR):
            shutil.rmtree(LOG_DIR)
        tf.summary.FileWriter(LOG_DIR, SESS.graph)

    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job) # 添加一个工作线程
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)  # tf 的线程调度

    np.save(reward_path, np.array(GLOBAL_RUNNING_R))

    save_count=0
    # if GLOBAL_EP % 1000 == 0 and save_path is not None:
    #     GLOBAL_AC.saver.save(SESS, os.path.join(os.getcwd(),save_path + '/' + str(save_count) + '/model-' + str(GLOBAL_EP) + '.ckpt'))
    #     if GLOBAL_EP % 5000 == 0:
    #         save_count += 1
    #         print("Saved Model")
    # save_path_ = GLOBAL_AC.saver.save(SESS, os.path.join(os.getcwd(), save_path))
    save_path_ = GLOBAL_AC.saver.save(SESS, save_path+ '/' + str(save_count) + '/model-' + str(GLOBAL_EP) + '.ckpt')
    print("Model saved in file: %s" % save_path_)

    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('step')
    plt.ylabel('Total moving reward')
    plt.savefig(save_path + "/training.png")

    plt.show(block=False)
