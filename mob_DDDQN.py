from __future__ import division
from future import standard_library
standard_library.install_aliases()
import itertools
import scipy.misc
import tensorflow as tf
import tensorflow.contrib.slim as slim
from builtins import range
from past.utils import old_div
import MalmoPython
import os
import sys
import json
import random
import errno
import numpy as np
import matplotlib.pyplot as plt
import time

start=time.clock()
#######################################gridworld################################################################################
class gameOb():
    def __init__(self, coordinates, size, intensity, channel, reward, name):
        self.x = coordinates[0]
        self.y = coordinates[1]
        self.size = size
        self.intensity = intensity
        self.channel = channel
        self.reward = reward
        self.name = name


class gameEnv():
    def __init__(self, partial, size):
        self.sizeX = size
        self.sizeY = size
        self.actions = 4
        self.objects = []
        self.partial = partial
        # a = self.reset()
        # plt.imshow(a, interpolation="nearest")

    # def reset(self):
    #     self.objects = []
    #     hero = gameOb(self.newPosition(), 1, 1, 2, None, 'hero')
    #     self.objects.append(hero)
    #     bug = gameOb(self.newPosition(), 1, 1, 1, 1, 'goal')
    #     self.objects.append(bug)
    #     hole = gameOb(self.newPosition(), 1, 1, 0, -1, 'fire')
    #     self.objects.append(hole)
    #     bug2 = gameOb(self.newPosition(), 1, 1, 1, 1, 'goal')
    #     self.objects.append(bug2)
    #     hole2 = gameOb(self.newPosition(), 1, 1, 0, -1, 'fire')
    #     self.objects.append(hole2)
    #     bug3 = gameOb(self.newPosition(), 1, 1, 1, 1, 'goal')
    #     self.objects.append(bug3)
    #     bug4 = gameOb(self.newPosition(), 1, 1, 1, 1, 'goal')
    #     self.objects.append(bug4)
    #     state = self.renderEnv()
    #     self.state = state
    #     return state

    def moveChar(self, direction):
        # 0 - up, 1 - down, 2 - left, 3 - right
        hero = self.objects[0]
        heroX = hero.x
        heroY = hero.y
        penalize = 0.
        if direction == 0 and hero.y >= 1:
            hero.y -= 1
        if direction == 1 and hero.y <= self.sizeY - 2:
            hero.y += 1
        if direction == 2 and hero.x >= 1:
            hero.x -= 1
        if direction == 3 and hero.x <= self.sizeX - 2:
            hero.x += 1
        if hero.x == heroX and hero.y == heroY:
            penalize = 0.0
        self.objects[0] = hero
        return penalize

    # def newPosition(self):
    #     iterables = [range(self.sizeX), range(self.sizeY)]
    #     points = []
    #     for t in itertools.product(*iterables):
    #         points.append(t)
    #     currentPositions = []
    #     for objectA in self.objects:
    #         if (objectA.x, objectA.y) not in currentPositions:
    #             currentPositions.append((objectA.x, objectA.y))
    #     for pos in currentPositions:
    #         points.remove(pos)
    #     location = np.random.choice(range(len(points)), replace=False)
    #     return points[location]

    def checkGoal(self):
        others = []
        for obj in self.objects:
            if obj.name == 'hero':
                hero = obj
            else:
                others.append(obj)
        ended = False
        for other in others:
            if hero.x == other.x and hero.y == other.y:
                self.objects.remove(other)
                if other.reward == 1:
                    self.objects.append(gameOb(self.newPosition(), 1, 1, 1, 1, 'goal'))
                else:
                    self.objects.append(gameOb(self.newPosition(), 1, 1, 0, -1, 'mob'))
                return other.reward, False
        if ended == False:
            return 0.0, False

    def renderEnv(self):
        # a = np.zeros([self.sizeY,self.sizeX,3])
        a = np.ones([self.sizeY + 2, self.sizeX + 2, 3])  # sizeY+2 rows, [self.sizeX+2,3] per row.
        # print("a.shape initial:", a.shape)
        a[1:-1, 1:-1, :] = 0
        hero = None
        for item in self.objects:
            # print(item.name)
            a[item.y + 1:item.y + item.size + 1, item.x + 1:item.x + item.size + 1, item.channel] = item.intensity
            if item.name == 'hero':
                hero = item
        if self.partial == True:
            a = a[hero.y:hero.y + 3, hero.x:hero.x + 3, :]
        b = scipy.misc.imresize(a[:, :, 0], [84, 84, 1], interp='nearest')
        c = scipy.misc.imresize(a[:, :, 1], [84, 84, 1], interp='nearest')
        d = scipy.misc.imresize(a[:, :, 2], [84, 84, 1], interp='nearest')
        a = np.stack([b, c, d], axis=2)
        # print("a.shape:",a.shape)
        return a

    # def step(self, action):
    #
    #     penalty = self.moveChar(action)
    #     reward, done = self.checkGoal()
    #     state = self.renderEnv()
    #     if reward == None:
    #         print(done)
    #         print(reward)
    #         print(penalty)
    #         return state, (reward + penalty), done
    #     else:
    #         return state, (reward + penalty), done


##########################################DDDON##################################################################################

env = gameEnv(partial=False,size=60)


class Qnetwork():
    def __init__(self, h_size):
        # The network recieves a frame from the game, flattened into an array.
        # It then resizes it and processes it through four convolutional layers.
        self.scalarInput = tf.placeholder(shape=[None, 21168], dtype=tf.float32)
        self.imageIn = tf.reshape(self.scalarInput, shape=[-1, 84, 84, 3])
        self.conv1 = slim.conv2d(inputs=self.imageIn, num_outputs=32, kernel_size=[8, 8], stride=[4, 4], padding='VALID',
            biases_initializer=None)
        self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs=64, kernel_size=[4, 4], stride=[2, 2], padding='VALID',
            biases_initializer=None)
        self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs=64, kernel_size=[3, 3], stride=[1, 1], padding='VALID',
            biases_initializer=None)
        self.conv4 = slim.conv2d(inputs=self.conv3, num_outputs=h_size, kernel_size=[7, 7], stride=[1, 1], padding='VALID',
            biases_initializer=None)

        # We take the output from the final convolutional layer and split it into separate advantage and value streams.
        self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)
        self.streamA = slim.flatten(self.streamAC)
        self.streamV = slim.flatten(self.streamVC)
        xavier_init = tf.contrib.layers.xavier_initializer()
        self.AW = tf.Variable(xavier_init([h_size // 2, env.actions]))
        self.VW = tf.Variable(xavier_init([h_size // 2, 1]))
        self.Advantage = tf.matmul(self.streamA, self.AW)
        self.Value = tf.matmul(self.streamV, self.VW)

        # Then combine them together to get our final Q-values.
        self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keepdims=True))
        self.predict = tf.argmax(self.Qout, 1)

        # Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
        self.targetQ = tf.placeholder(shape=[None], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None], dtype=tf.int32)
        self.actions_onehot = tf.one_hot(self.actions, env.actions, dtype=tf.float32)

        self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot), axis=1)

        self.td_error = tf.square(self.targetQ - self.Q)
        self.loss = tf.reduce_mean(self.td_error)
        self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
        self.updateModel = self.trainer.minimize(self.loss)

# Experience Replay
# This class allows us to store experies and sample then randomly to train the network.
class experience_buffer():
    def __init__(self, buffer_size=50000):
        self.buffer = []
        self.buffer_size = buffer_size

    def add(self, experience):
        if len(self.buffer) + len(experience) >= self.buffer_size:
            self.buffer[0:(len(experience) + len(self.buffer)) - self.buffer_size] = []
        self.buffer.extend(experience)

    def sample(self, size):
        # print("self.buffer:",self.buffer)
        return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])  # size=32
        # return np.reshape(np.array(random.sample(self.buffer, size)), [size, 60])

# This is a simple function to resize our game frames.
def processState(states):
    return np.reshape(states,[21168])

# These functions allow us to update the parameters of our target network with those of the primary network.
def updateTargetGraph(tfVars,tau):
    total_vars = len(tfVars)
    op_holder = []
    for idx,var in enumerate(tfVars[0:total_vars//2]):
        op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
    return op_holder

def updateTarget(op_holder,sess):
    for op in op_holder:
        sess.run(op)

# Training the network
# Setting all the training parameters
batch_size = 32 #How many experiences to use for each training step.
update_freq = 4 #How often to perform a training step.
y = .99 #Discount factor on the target Q-values
startE = 1 #Starting chance of random action
endE = 0.1 #Final chance of random action
annealing_steps = 10000. #How many steps of training to reduce startE to endE.
num_episodes = 10000 #How many episodes of game environment to train network with.
pre_train_steps = 10000 #How many steps of random actions before training begins.
max_epLength = 50 #The max allowed length of our episode.
load_model = False #Whether to load a saved model.
path = "./dqn" #The path to save our model to.
h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 #Rate to update target network toward primary network

tf.reset_default_graph()
mainQN = Qnetwork(h_size)
targetQN = Qnetwork(h_size)

init = tf.global_variables_initializer()

saver = tf.train.Saver()

trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables, tau)

myBuffer = experience_buffer()

# Set the rate of random action decrease.
epsilon = startE
stepDrop = (startE - endE) / annealing_steps

# create lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0

# Make a path for our model to be saved in.
if not os.path.exists(path):
    os.makedirs(path)

#############################################################################################################################
############################################GameEnvironment##################################################################
if sys.version_info[0] == 2:
    # Workaround for https://github.com/PythonCharmers/python-future/issues/262
    import Tkinter as tk
else:
    import tkinter as tk
from collections import namedtuple
EntityInfo = namedtuple('EntityInfo', 'x, y, z, yaw, pitch, name, colour, variation, quantity, life')
EntityInfo.__new__.__defaults__ = (0, 0, 0, 0, 0, "", "", "", 1, "")

# Task parameters:
NUM_GOALS = 20
NUM_MOBS=50
AGENT_TYPE = "The Hunted"
GOAL_TYPE = "apple"
GOAL_REWARD = 1
MOB_REWARD = -1
ARENA_WIDTH = 60
ARENA_BREADTH = 60
MOB_TYPE = "Endermite"  # Change for fun, but note that spawning conditions have to be correct - eg spiders will require darker conditions.
action_space = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]
n_actions = len(action_space)

# Display parameters:
CANVAS_BORDER = 20
CANVAS_WIDTH = 400
CANVAS_HEIGHT = CANVAS_BORDER + ((CANVAS_WIDTH - CANVAS_BORDER) * ARENA_BREADTH / ARENA_WIDTH)
CANVAS_SCALEX = old_div((CANVAS_WIDTH-CANVAS_BORDER),ARENA_WIDTH)
CANVAS_SCALEY = old_div((CANVAS_HEIGHT-CANVAS_BORDER),ARENA_BREADTH)
CANVAS_ORGX = old_div(-ARENA_WIDTH,CANVAS_SCALEX)
CANVAS_ORGY = old_div(-ARENA_BREADTH,CANVAS_SCALEY)

# Agent parameters:
agent_stepsize = 1
agent_search_resolution = 30 # Smaller values make computation faster, which seems to offset any benefit from the higher resolution.
agent_goal_weight = 100
agent_edge_weight = -100
agent_mob_weight = -10
agent_turn_weight = 0 # Negative values to penalise turning, positive to encourage.

def getItemXML():
    ''' Build an XML string that contains some randomly positioned goal items'''
    xml=""
    for item in range(NUM_GOALS):
        x = str(random.randint(old_div(-ARENA_WIDTH,2),old_div(ARENA_WIDTH,2)))
        z = str(random.randint(old_div(-ARENA_BREADTH,2),old_div(ARENA_BREADTH,2)))
        xml += '''<DrawItem x="''' + x + '''" y="210" z="''' + z + '''" type="''' + GOAL_TYPE + '''"/>'''
    return xml

def getCorner(index,top,left,expand=0,y=206):
    ''' Return part of the XML string that defines the requested corner'''
    x = str(-(expand+old_div(ARENA_WIDTH,2))) if left else str(expand+old_div(ARENA_WIDTH,2))
    z = str(-(expand+old_div(ARENA_BREADTH,2))) if top else str(expand+old_div(ARENA_BREADTH,2))
    return 'x'+index+'="'+x+'" y'+index+'="' +str(y)+'" z'+index+'="'+z+'"'

def getMissionXML(summary):
    ''' Build an XML mission string.'''
    spawn_end_tag = ' type="mob_spawner" variant="' + MOB_TYPE + '"/>'
    return '''<?xml version="1.0" encoding="UTF-8" ?>
    <Mission xmlns="http://ProjectMalmo.microsoft.com" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
        <About>
            <Summary>''' + summary + '''</Summary>
        </About>
        <ModSettings>
            <MsPerTick>20</MsPerTick>
        </ModSettings>
        <ServerSection>
            <ServerInitialConditions>
                <Time>
                    <StartTime>13000</StartTime>
                    <AllowPassageOfTime>false</AllowPassageOfTime>
                </Time>
                <AllowSpawning>true</AllowSpawning>
                <AllowedMobs>''' + MOB_TYPE + '''</AllowedMobs>
            </ServerInitialConditions>
            <ServerHandlers>
                <FlatWorldGenerator generatorString="3;7,220*1,5*3,2;3;,biome_1" />
                <DrawingDecorator>
                    <DrawCuboid ''' + getCorner("1",True,True,expand=1) + " " + getCorner("2",False,False,y=226,expand=1) + ''' type="stone"/>
                    <DrawCuboid ''' + getCorner("1",True,True,y=207) + " " + getCorner("2",False,False,y=226) + ''' type="air"/>
                    <DrawLine ''' + getCorner("1",True,True) + " " + getCorner("2",True,False) + spawn_end_tag + '''
                    <DrawLine ''' + getCorner("1",True,True) + " " + getCorner("2",False,True) + spawn_end_tag + '''
                    <DrawLine ''' + getCorner("1",False,False) + " " + getCorner("2",True,False) + spawn_end_tag + '''
                    <DrawLine ''' + getCorner("1",False,False) + " " + getCorner("2",False,True) + spawn_end_tag + '''
                    <DrawCuboid x1="-1" y1="206" z1="-1" x2="1" y2="206" z2="1" ''' + spawn_end_tag + '''
                    ''' + getItemXML() + '''
                </DrawingDecorator>
                <ServerQuitWhenAnyAgentFinishes description="achieve_goal"/>
            </ServerHandlers>
        </ServerSection>
        <AgentSection mode="Survival">
            <Name>'''+AGENT_TYPE+'''</Name>
            <AgentStart>
                <Placement x="0.5" y="207.0" z="0.5"/>
                <Inventory>
                </Inventory>
            </AgentStart>
            <AgentHandlers>
                <ChatCommands/>
                <DiscreteMovementCommands/>
                <ObservationFromNearbyEntities>
                    <Range name="entities" xrange="'''+str(ARENA_WIDTH)+'''" yrange="2" zrange="'''+str(ARENA_BREADTH)+'''" />
                </ObservationFromNearbyEntities>
                <ObservationFromFullStats/>
                <RewardForCollectingItem>
                    <Item type="'''+GOAL_TYPE+'''" reward="'''+str(GOAL_REWARD)+'''"/>
                </RewardForCollectingItem>
                <RewardForMissionEnd rewardForDeath="-3">
                    <Reward description="achieve_goal" reward="0" />
                </RewardForMissionEnd>
                <VideoProducer viewpoint="2">
                    <Width>860</Width>
                    <Height>480</Height>
                </VideoProducer>
            </AgentHandlers>
        </AgentSection>
    </Mission>'''

recordingsDirectory="FleeRecordings"
try:
    os.makedirs(recordingsDirectory)
except OSError as exception:
    if exception.errno != errno.EEXIST: # ignore error if already existed
        raise

if sys.version_info[0] == 2:
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)  # flush print output immediately
else:
    import functools
    print = functools.partial(print, flush=True)

root = tk.Tk()
root.wm_title("Collect the " + GOAL_TYPE + "s, dodge the " + MOB_TYPE + "s!")

canvas = tk.Canvas(root, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, borderwidth=0, highlightthickness=0, bg="black")
canvas.pack()
root.update()

def canvasX(x):
    return (old_div(CANVAS_BORDER,2)) + (0.5 + old_div(x,float(ARENA_WIDTH))) * (CANVAS_WIDTH-CANVAS_BORDER)

def canvasY(y):
    return (old_div(CANVAS_BORDER,2)) + (0.5 + old_div(y,float(ARENA_BREADTH))) * (CANVAS_HEIGHT-CANVAS_BORDER)

def drawMobs(entities, flash):
    canvas.delete("all")
    if flash:
        canvas.create_rectangle(0,0,CANVAS_WIDTH,CANVAS_HEIGHT,fill="#ff0000") # Pain.
    canvas.create_rectangle(canvasX(old_div(-ARENA_WIDTH,2)), canvasY(old_div(-ARENA_BREADTH,2)), canvasX(old_div(ARENA_WIDTH,2)), canvasY(old_div(ARENA_BREADTH,2)), fill="#888888")
    for ent in entities:
        if ent.name == MOB_TYPE:
            canvas.create_oval(canvasX(ent.x)-2, canvasY(ent.z)-2, canvasX(ent.x)+2, canvasY(ent.z)+2, fill="#ff2244")
        elif ent.name == GOAL_TYPE:
            canvas.create_oval(canvasX(ent.x)-3, canvasY(ent.z)-3, canvasX(ent.x)+3, canvasY(ent.z)+3, fill="#4422ff")
        else:
            yaw = ent.yaw
            # canvas.create_arc(canvasX(ent.x)-4, canvasY(ent.z)-4, canvasX(ent.x)+4, canvasY(ent.z)+4, fill="#22ff44")
            canvas.create_oval(canvasX(ent.x)-4, canvasY(ent.z)-4, canvasX(ent.x)+4, canvasY(ent.z)+4, fill="#22ff44")
    root.update()

validate = True

my_client_pool = MalmoPython.ClientPool()
my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10000))
my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10001))
my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10002))
my_client_pool.add(MalmoPython.ClientInfo("127.0.0.1", 10003))

agent_host = MalmoPython.AgentHost()
try:
    agent_host.parse( sys.argv )
except RuntimeError as e:
    print('ERROR:',e)
    print(agent_host.getUsage())
    exit(1)
if agent_host.receivedArgument("help"):
    print(agent_host.getUsage())
    exit(0)

if agent_host.receivedArgument("test"):
    num_reps = 1
else:
    num_reps = 100

def get_ob(msg):
    ob = json.loads(msg)
    env.objects=[]
    if "entities" in ob:
        for entity in ob[u'entities']:
            if entity["name"]==AGENT_TYPE:
                hero = gameOb((int(round(entity["x"]+30)),int(round(entity["z"]+30))), 1, 1, 2, None, 'hero')
                env.objects.append(hero)
            if entity["name"] == GOAL_TYPE:
                goal = gameOb((int(round(entity["x"]+30)),int(round(entity["z"]+30))), 1, 1, 1, 1, 'goal')
                env.objects.append(goal)
            if entity["name"] == MOB_TYPE:
                mob = gameOb((int(round(entity["x"]+30)), int(round(entity["z"]+30))), 1, 1, 0, -1, 'mob')
                env.objects.append(mob)
    state=env.renderEnv()
    return state

# RL take action and get next observation and reward
def step(s,agent_host,a,DamageTaken):
    # ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]
    s_=s
    reward = -3
    done = False
    agent_host.sendCommand(action_space[a])
    # print(action_space[a])
    time.sleep(0.1)
    world_state = agent_host.getWorldState()
    if world_state.is_mission_running:
        # get next observation and reward
        if world_state.number_of_observations_since_last_state > 0:
            reward=0.0
            msg = world_state.observations[-1].text
            ob = json.loads(msg)
            if "isAlive" in ob:
                if not ob[u'isAlive']:
                    done = True
            s_ = get_ob(msg)
            s_=processState(s_)
            # print("s_:",s_)

            try:
                # A reward signal has come in - see what it is:
                reward = world_state.rewards[-1].getValue()
            except:
                pass

            if "DamageTaken" in ob:
                try:
                    if ob[u'DamageTaken']>DamageTaken[-1]:
                       reward+=MOB_REWARD*(ob[u'DamageTaken']-DamageTaken[-1])/20
                       DamageTaken.append(ob[u'DamageTaken'])
                except:
                    pass
        else:
            done=True
    else:
        done=True

    return s_, reward,done,DamageTaken

###############################################################################################################
##########################################################################################################################
with tf.Session() as sess:
    sess.run(init)
    if load_model == True:
        print('Loading Model...')
        ckpt = tf.train.get_checkpoint_state(path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    current_life = 0
    all_ep_r = []
    reward_his =[]
    i = 0
    for ep in range(num_episodes):
        episodeBuffer=experience_buffer()
        mission_xml = getMissionXML(MOB_TYPE + " Apocalypse #" + str(ep))
        my_mission = MalmoPython.MissionSpec(mission_xml,validate)
        max_retries = 10
        for retry in range(max_retries):
            try:
                # Set up a recording
                my_mission_record = MalmoPython.MissionRecordSpec(recordingsDirectory + "//" + "Mission_" + str(ep) + ".tgz")
                my_mission_record.recordRewards()

                # Attempt to start the mission:
                agent_host.startMission( my_mission, my_client_pool,my_mission_record,0, "predatorExperiment" )
                break
            except RuntimeError as e:
                if retry == max_retries - 1:
                    print("Error starting mission",e)
                    print("Is the game running?")
                    exit(1)
                else:
                    time.sleep(2)

        world_state = agent_host.getWorldState()

        while not world_state.has_mission_begun:
            time.sleep(0.1)
            world_state = agent_host.getWorldState()
        while not world_state.number_of_observations_since_last_state > 0:
            time.sleep(0.02)
            world_state = agent_host.getWorldState()

        agent_host.sendCommand(action_space[1])#"move 1")    # run!
        # print(action_space[1])

        # main loop:
        total_reward = 0
        total_commands = 0
        flash = False
        ep_r = 0

        # print(world_state.is_mission_running)
        msg = world_state.observations[-1].text
        ob = json.loads(msg)
        # s = get_ob(msg)
        # s= processState(s)
        # print("initial s:",s)

        done=False
        rAll=0
        j=0

        DamageTaken = []
        if "DamageTaken" in ob:
            DamageTaken.append(ob[u'DamageTaken'])

        while world_state.is_mission_running:  # in one episode
            world_state = agent_host.getWorldState()
            if world_state.number_of_observations_since_last_state > 0:
                msg = world_state.observations[-1].text
                s = processState(get_ob(msg))
                ob = json.loads(msg)
                if "Life" in ob:
                    life = ob[u'Life']
                    if life < current_life:
                        # agent_host.sendCommand("chat aaaaaaaaargh!!!")
                        flash = True
                    current_life = life
                if "entities" in ob:
                    entities = [EntityInfo(**k) for k in ob["entities"]]
                    drawMobs(entities, flash)

                    if np.random.rand(1) < epsilon or total_steps < pre_train_steps:
                        a = np.random.randint(0, 4)
                    else:
                        # print("s.shape:",s.shape)
                        a = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: [s]})[0]

                    # take action and get next observation and reward
                    s_, r,d, DamageTaken = step(s,agent_host, a, DamageTaken)
                    print("action:",a,"reward:",r)
                    # print("s_:",s_)
                    # for ss in s_:
                    #     if ss!=255:
                    #         print(ss)

                    total_steps += 1
                    episodeBuffer.add(np.reshape(np.array([s, a, r, s_, d]),[1, 5]))  # Save the experience to our episode buffer.

                    if total_steps > pre_train_steps:
                        if epsilon > endE:
                            epsilon -= stepDrop

                        if total_steps % (update_freq) == 0:
                            trainBatch = myBuffer.sample(batch_size)  # Get a random batch of experiences.
                            # Below we perform the Double-DQN update to the target Q-values
                            Q1 = sess.run(mainQN.predict, feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 3])})
                            Q2 = sess.run(targetQN.Qout, feed_dict={targetQN.scalarInput: np.vstack(trainBatch[:, 3])})
                            end_multiplier = -(trainBatch[:, 4] - 1)
                            doubleQ = Q2[range(batch_size), Q1]
                            targetQ = trainBatch[:, 2] + (y * doubleQ * end_multiplier)
                            # Update the network with our target values.
                            _ = sess.run(mainQN.updateModel, feed_dict={mainQN.scalarInput: np.vstack(trainBatch[:, 0]),
                                                    mainQN.targetQ: targetQ, mainQN.actions: trainBatch[:, 1]})

                            updateTarget(targetOps, sess)  # Update the target network toward the primary network.
                    rAll += r
                    s = s_

                    if done == True:
                        break
            time.sleep(0.02)
            flash = False

        myBuffer.add(episodeBuffer.buffer)
        jList.append(j)
        rList.append(rAll)
        i+=1

        print("game:",i,"rAll:",rAll,"Time used:",(time.clock() - start))
        # Periodically save the model.
        if i % 1000 == 0:
            saver.save(sess, path + '/model-' + str(i) + '.ckpt')
            print("Saved Model")
        if i% 200 ==0:
            plt.close('all')
            rMat = np.resize(np.array(rList), [len(rList) // 100, 100])
            rMean = np.average(rMat, 1)
            plt.ylabel('Reward')
            plt.xlabel('Training Steps')
            plt.plot(rMean)
            plt.show()
        if len(rList) % 10 == 0:
            print(total_steps, np.mean(rList[-10:]), epsilon)


    for error in world_state.errors:
        print("Error:", error.text)
    time.sleep(1)  # Give the mod a little time to prepare for the next mission.

    saver.save(sess, path + '/model-' + str(i) + '.ckpt')
print("Percent of succesful episodes: " + str(sum(rList) / num_episodes) + "%")

plt.close('all')
rMat = np.resize(np.array(rList), [len(rList) // 100, 100])
rMean = np.average(rMat, 1)
plt.ylabel('Reward')
plt.xlabel('Training Steps')
plt.plot(rMean)
plt.show()