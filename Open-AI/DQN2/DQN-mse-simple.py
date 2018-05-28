# OpenGym CartPole-v0
# -------------------
#
# This code demonstrates use a full DQN implementation
# to solve OpenGym CartPole-v0 problem.
#
# Made as part of blog series Let's make a DQN, available at:
# https://jaromiru.com/2016/09/27/lets-make-a-dqn-theory/
#
# author: Jaromir Janisch, 2016

import random, numpy, math, gym, sys
from keras import backend as K

import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

tuning = [
    {
        'lr': 0.0001,
        'memory': 500000,
        'batch': 512,
        'gamma': 0.99,
        'max_eps': 1,
        'min_eps': 0.1,
        'lambda': 0.0001,
        'utf': 1000,
        'uf': 10,
        'hlu': 50,
        'output_dir': './dqn-small-local1',
        'output_name': 'LunarLander-DQN-PER'
    },
    {
        'lr': 0.0001,
        'memory': 500000,
        'batch': 256,
        'gamma': 0.99,
        'max_eps': 1,
        'min_eps': 0.1,
        'lambda': 0.0001,
        'utf': 1000,
        'uf': 10,
        'hlu': 64,
        'output_dir': './dqn-small-local2',
        'output_name': 'LunarLander-DQN-PER'
    }
]

# ----------
tune_id = int(sys.argv[1])

LEARNING_RATE = tuning[tune_id]['lr']

# -------------------- AGENT ---------------------------
MEMORY_CAPACITY = tuning[tune_id]['memory']
BATCH_SIZE = tuning[tune_id]['batch']

GAMMA = tuning[tune_id]['gamma']

MAX_EPSILON = tuning[tune_id]['max_eps']
MIN_EPSILON = tuning[tune_id]['min_eps']
LAMBDA = tuning[tune_id]['lambda']  # speed of decay

UPDATE_TARGET_FREQUENCY = tuning[tune_id]['utf']
UPDATE_FREQUENCY = tuning[tune_id]['uf']   # how often to replay batch and train
TRAINING_EPISODES = 50000
HIDDEN_LAYER_UNITS = tuning[tune_id]['hlu']

OUTPUT_DIR = tuning[tune_id]['output_dir']
import os
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)
OUTPUT_NAME = tuning[tune_id]['output_name']
file = open(OUTPUT_DIR + '/' + OUTPUT_NAME + '.txt', 'w+')


# -------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *


class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        self.model_ = self._createModel()

    def _createModel(self):
        model = Sequential()

        model.add(Dense(units=HIDDEN_LAYER_UNITS, activation='relu', input_dim=stateCnt))
        # model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=actionCnt, activation='linear'))

        opt = Adam(lr=LEARNING_RATE)
        model.compile(loss='mean_squared_error', optimizer=opt)

        return model

    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=BATCH_SIZE, epochs=epochs, verbose=verbose)

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, self.stateCnt), target=target).flatten()

    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())


# -------------------- MEMORY --------------------------
class Memory:  # stored as ( s, a, r, s_ )
    samples = []

    def __init__(self, capacity):
        self.capacity = capacity

    def add(self, sample):
        self.samples.append(sample)

        if len(self.samples) > self.capacity:
            self.samples.pop(0)

    def sample(self, n):
        n = min(n, len(self.samples))
        return random.sample(self.samples, n)

    def isFull(self):
        return len(self.samples) >= self.capacity


class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)
        self.memory = Memory(MEMORY_CAPACITY)

    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt - 1)
        else:
            return numpy.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()

        # debug the Q function in poin S
        # if self.steps % 100 == 0:
        #     S = numpy.array([-0.01335408, -0.04600273, -0.00677248, 0.01517507])
        #     pred = agent.brain.predictOne(S)
        #     print(pred[0])
        #     sys.stdout.flush()

        # slowly decrease Epsilon based on our experience
        self.steps += 1
        if abs(self.epsilon - MIN_EPSILON) < 0.0001:
            self.epsilon = MIN_EPSILON
        else:
            self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)

    def replay(self):
        batch = self.memory.sample(BATCH_SIZE)
        batchLen = len(batch)

        no_state = numpy.zeros(self.stateCnt)

        states = numpy.array([o[0] for o in batch])
        states_ = numpy.array([(no_state if o[3] is None else o[3]) for o in batch])

        p = self.brain.predict(states)
        p_ = self.brain.predict(states_, target=True)

        x = numpy.zeros((batchLen, self.stateCnt))
        y = numpy.zeros((batchLen, self.actionCnt))

        for i in range(batchLen):
            o = batch[i]
            s, a, r, s_ = o

            t = p[i]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * numpy.amax(p_[i])

            x[i], y[i] = s, t

        self.brain.train(x, y)


class RandomAgent:
    memory = Memory(MEMORY_CAPACITY)

    def __init__(self, actionCnt):
        self.actionCnt = actionCnt

    def act(self, s):
        return random.randint(0, self.actionCnt - 1)

    def observe(self, sample):  # in (s, a, r, s_) format
        self.memory.add(sample)

    def replay(self):
        pass


# -------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)
        self.index = 0

    def run(self, agent):
        s = self.env.reset()
        R = 0
        steps = 0
        while True:
            # self.env.render()
            steps += 1

            a = agent.act(s)

            s_, r, done, info = self.env.step(a)

            if done:  # terminal state
                s_ = None

            agent.observe((s, a, r, s_))
            if steps % UPDATE_FREQUENCY == 0:
                agent.replay()

            s = s_
            R += r

            if done:
                break
        if isinstance(agent, Agent):
            rList.append(R)
            stepList.append(steps)
            lenRList = len(rList)
            print('Episode ' + str(lenRList) + " reward " + str(R) + ' in ' + str(steps) + ' steps, epsilon=' + str(agent.epsilon))
            file.write('Episode ' + str(lenRList) + " reward " + str(R) + ' in ' + str(steps) + ' steps, epsilon=' + str(agent.epsilon))
            file.write('\n')
            if lenRList % 100 == 0:
                print("Drawing plot")
                plt.close('all')
                rMat = np.resize(np.array(rList), [len(rList) // 100, 100])
                rMean = np.average(rMat, 1)
                plt.plot(rMean)
                plt.xlabel('Training episodes (hundreds)')
                plt.ylabel('Average rewards every 100 episodes')
                plt.savefig(OUTPUT_DIR + "/" + OUTPUT_NAME + ".png")

                # if lenRList % 1000 == 0:
                self.index += 1
                print("Saving model")
                # Save memory model
                agent.brain.model.save(OUTPUT_DIR + "/" + OUTPUT_NAME + str(self.index) + ".h5")
                # Save reward list
                np.save(OUTPUT_DIR + "/" + OUTPUT_NAME + str(self.index) + "-rList", rList)
                np.save(OUTPUT_DIR + "/" + OUTPUT_NAME + str(self.index) + "-stepList", stepList)


# -------------------- MAIN ----------------------------
PROBLEM = 'LunarLander-v2'
env = Environment(PROBLEM)

stateCnt = env.env.observation_space.shape[0]
actionCnt = env.env.action_space.n

agent = Agent(stateCnt, actionCnt)
randomAgent = RandomAgent(actionCnt)

rList = []
stepList = []

try:
    while randomAgent.memory.isFull() == False:
        env.run(randomAgent)

    agent.memory.samples = randomAgent.memory.samples
    randomAgent = None

    for _ in range(TRAINING_EPISODES):
        env.run(agent)
finally:
    agent.brain.model.save(OUTPUT_DIR + "/" + OUTPUT_NAME + "-final.h5")
    file.close()