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
import keras
from keras import backend as K

import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from SumTree import SumTree

tuning = [
    {
        'lr': 0.0001,
        'memory': 500000,
        'batch': 512,
        'gamma': 0.99,
        'max_eps': 1,
        'min_eps': 0.1,
        'annealing_steps': 10000,
        'utf': 1000,
        'uf': 10,
        'hlu': 50,
        'output_dir': './dqn-per-final',
        'output_name': 'LunarLander-DQN-PER'
    },
    {
        'lr': 0.0001,
        'memory': 500000,
        'batch': 512,
        'gamma': 0.99,
        'max_eps': 1,
        'min_eps': 0.1,
        'annealing_steps': 10000,
        'utf': 1000,
        'uf': 10,
        'hlu': 50,
        'output_dir': './dqn-per-final2',
        'output_name': 'LunarLander-DQN-PER'
    },
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
ANNEALING_STEPS = tuning[tune_id]['annealing_steps']  # speed of decay

UPDATE_TARGET_FREQUENCY = tuning[tune_id]['utf']
UPDATE_FREQUENCY = tuning[tune_id]['uf']   # how often to replay batch and train
TRAINING_EPISODES = 50000
HIDDEN_LAYER_UNITS = tuning[tune_id]['hlu']

OUTPUT_DIR = tuning[tune_id]['output_dir']

OUTPUT_NAME = tuning[tune_id]['output_name']



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
        self.history = LossHistory()
        self.loaded = False

    def _createModel(self):
        model = Sequential()

        model.add(Dense(units=HIDDEN_LAYER_UNITS, activation='relu', input_dim=stateCnt))
        # model.add(Dense(units=64, activation='relu'))
        model.add(Dense(units=actionCnt, activation='linear'))

        opt = Adam(lr=LEARNING_RATE)
        model.compile(loss='mean_squared_error', optimizer=opt)

        return model

    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=BATCH_SIZE, epochs=epochs, verbose=verbose, callbacks=[self.history])
        lossesList.extend(self.history.losses)

    def predict(self, s, target=False):
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        return self.predict(s.reshape(1, self.stateCnt), target=target).flatten()

    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())

    def load(self, partno):
        del self.model
        if partno > 0:
            filepath = OUTPUT_DIR + '/' + OUTPUT_NAME + str(partno) + '.h5'
        else:
            filepath = OUTPUT_DIR + '/' + OUTPUT_NAME + '-final.h5'
        self.model = keras.models.load_model(filepath)
        self.loaded = True

    def isLoaded(self):
        return self.loaded


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample)

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)


class Agent:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt
        self.steps = 0
        self.epsilon = MAX_EPSILON

        self.brain = Brain(stateCnt, actionCnt)
        # self.memory = Memory(MEMORY_CAPACITY)

    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt - 1)
        else:
            return numpy.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        _, _, errors = self._getTargets([(0, sample)])
        self.memory.add(errors[0], sample)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()

        # slowly decrease Epsilon based on our eperience
        self.steps += 1

        if self.steps < ANNEALING_STEPS:
            self.epsilon = MAX_EPSILON - (MAX_EPSILON - MIN_EPSILON) / ANNEALING_STEPS * self.steps
        else:
            self.epsilon = MIN_EPSILON

    def _getTargets(self, batch):
        no_state = numpy.zeros(self.stateCnt)
        states = numpy.array([o[1][0] for o in batch])
        states_ = numpy.array([(no_state if o[1][3] is None else o[1][3]) for o in batch])

        policies = agent.brain.predict(states)
        pTarget_ = agent.brain.predict(states_, target=True)

        x = numpy.zeros((len(batch), self.stateCnt))
        y = numpy.zeros((len(batch), self.actionCnt))
        errors = numpy.zeros(len(batch))

        for i in range(len(batch)):
            o = batch[i][1]
            s, a, r, s_ = o

            t = policies[i]

            oldVal = t[a]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * numpy.amax(pTarget_[i])

            x[i] = s
            y[i] = t
            errors[i] = abs(oldVal - t[a])

        return (x, y, errors)

    def replay(self):
        batch = self.memory.sample(BATCH_SIZE)
        x, y, errors = self._getTargets(batch)

        # update errors
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])

        self.brain.train(x, y)

        # batch = self.memory.sample(BATCH_SIZE)
        # batchLen = len(batch)
        #
        # no_state = numpy.zeros(self.stateCnt)
        #
        # states = numpy.array([o[0] for o in batch])
        # states_ = numpy.array([(no_state if o[3] is None else o[3]) for o in batch])
        #
        # p = self.brain.predict(states)
        # p_ = self.brain.predict(states_, target=True)
        #
        # x = numpy.zeros((batchLen, self.stateCnt))
        # y = numpy.zeros((batchLen, self.actionCnt))
        #
        # for i in range(batchLen):
        #     o = batch[i]
        #     s, a, r, s_ = o
        #
        #     t = p[i]
        #     if s_ is None:
        #         t[a] = r
        #     else:
        #         t[a] = r + GAMMA * numpy.amax(p_[i])
        #
        #     x[i], y[i] = s, t
        #
        # self.brain.train(x, y)


class RandomAgent:
    memory = Memory(MEMORY_CAPACITY)
    exp = 0

    def __init__(self, actionCnt):
        self.actionCnt = actionCnt

    def act(self, s):
        return random.randint(0, self.actionCnt - 1)

    def observe(self, sample):  # in (s, a, r, s_) format
        error = abs(sample[2])  # reward
        self.memory.add(error, sample)
        self.exp += 1

    def replay(self):
        pass


# -------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)
        self.index = 0

    def experiment(self, agent):
        agent.epsilon = 0
        rewards = []
        for i in range(100):
            s = self.env.reset()
            R = 0
            done = False
            while not done:
                a = agent.act(s)
                s_, r, d, _ = self.env.step(a)
                s = s_
                R += r
                done = d

            rewards.append(R)
            print('experiment #' + str(i+1) + ': ' + str(R))
        print('Average reward:', np.average(rewards))
        np.save(OUTPUT_DIR + "/experiment_results", rewards)

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
            stepsList.append(steps)

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
                np.save(OUTPUT_DIR + "/" + OUTPUT_NAME + str(self.index) + "-rewards", rList)
                np.save(OUTPUT_DIR + "/" + OUTPUT_NAME + str(self.index) + "-steps", stepsList)
                np.save(OUTPUT_DIR + "/" + OUTPUT_NAME + str(self.index) + "-losses", lossesList)




# -------------------- MAIN ----------------------------
if __name__ == "__main__":
    PROBLEM = 'LunarLander-v2'
    env = Environment(PROBLEM)

    stateCnt = env.env.observation_space.shape[0]
    actionCnt = env.env.action_space.n

    agent = Agent(stateCnt, actionCnt)
    randomAgent = RandomAgent(actionCnt)

    rList = []
    stepsList = []
    lossesList = []

    if len(sys.argv) == 3:
        partno = int(sys.argv[2])
        agent.brain.load(partno)
        env.experiment(agent)
    else:
        try:
            import os

            if not os.path.exists(OUTPUT_DIR):
                os.makedirs(OUTPUT_DIR)
            file = open(OUTPUT_DIR + '/' + OUTPUT_NAME + '.txt', 'w+')
            print("Initialization with random agent...")
            while randomAgent.exp < MEMORY_CAPACITY:
                env.run(randomAgent)
                # print(randomAgent.exp, "/", MEMORY_CAPACITY)

            agent.memory = randomAgent.memory

            randomAgent = None

            print("Starting learning")
            for _ in range(TRAINING_EPISODES):
                env.run(agent)
        finally:
            file.close()
            agent.brain.model.save(OUTPUT_DIR + "/" + OUTPUT_NAME + "-final.h5")