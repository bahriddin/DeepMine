from __future__ import print_function
from __future__ import division

# ------------------------------------------------------------------------------------------------
# Copyright (c) 2016 Microsoft Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute,
# sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
# NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
# DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
# ------------------------------------------------------------------------------------------------

# Demo of mob_spawner block - creates an arena, lines it with mob spawners of a given type, and then tries to keep an agent alive.
# Just a bit of fun - no real AI in here!

import tensorflow as tf
from future import standard_library
standard_library.install_aliases()
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
NUM_MOBS=100
AGENT_TYPE = "The Hunted"
GOAL_TYPE = "apple"
GOAL_REWARD = 1
MOB_REWARD = -1
ARENA_WIDTH = 60
ARENA_BREADTH = 60
MOB_TYPE = "Endermite"  # Change for fun, but note that spawning conditions have to be correct - eg spiders will require darker conditions.

# a=[i for i in range(0,190,10)]
# b=[-i for i in range(10,180,10)]
# action_space = [ str(i) for i in a+b ]

# action_space=np.arange(-1,1,0.1)
# print(action_space)
# print(len(action_space))

# # # action_space = ['0', '45','90', '135','180','225', '270','315']
# # # action_space = ["movenorth 1", "movesouth 1", "movewest 1", "moveeast 1"]
# n_actions = len(action_space)
n_actions=1
a_bound=np.array([1.])

# print(type(action_bound))
# action_bound=[-180,180]
# n_features = 3*ARENA_WIDTH*ARENA_BREADTH#7 + 2*NUM_GOALS + 2*NUM_MOBS
n_features = 5 + 2*NUM_GOALS + 2*NUM_MOBS
# print(n_features)

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


def get_ob(msg):
    observation = []
    ob = json.loads(msg)
    # print(ob)
    # print(type(ob))
    # print(type(ob.items))

    life=20.0
    food=20
    xpos=0.0
    zpos=0.0
    yaw=0.0
    features=[["Life",life],["Food",food],["Yaw",yaw],["XPos",xpos],["ZPos",zpos]]
    for i in range(len(features)):
        if features[i][0] in ob:
            features[i][1]=ob[features[i][0]]
    for feature in features:
        observation.append(feature[1])

    if "entities" in ob:
        goal_count = 0
        mob_count = 0
        goal_list=[]
        mob_list=[]
        for entity in ob[u'entities']:
            # if entity["name"]==AGENT_TYPE:
            #     # print(entity["name"])
            #     agent_yaw=entity["yaw"]
            #     agent_x=entity["x"]
            #     agent_z=entity["z"]
            if entity["name"] == GOAL_TYPE:
                # print(entity["name"])
                goal_list.append((entity["x"],entity["z"]))
                goal_count+=1
            if entity["name"] == MOB_TYPE:
                # print(entity["name"])
                mob_list.append((entity["x"],entity["z"]))
                mob_count += 1
        # observation.append(agent_yaw)
        # observation.append(agent_x)
        # observation.append(agent_z)

        if goal_count < NUM_GOALS:
            for goal in goal_list:
                observation.append(goal[0])
                observation.append(goal[1])
            for _ in range(NUM_GOALS - goal_count):
                observation.append(0.0)
                observation.append(0.0)
        else:
            for goal in goal_list[0:NUM_GOALS]:
                observation.append(goal[0])
                observation.append(goal[1])

        if mob_count<NUM_MOBS:
            for mob in mob_list:
                observation.append(mob[0])
                observation.append(mob[1])
            for _ in range(NUM_MOBS - mob_count):
                observation.append(0.0)
                observation.append(0.0)
        else:
            for mob in mob_list[:NUM_MOBS]:
                observation.append(mob[0])
                observation.append(mob[1])

    observation = np.array(observation)
    return observation

# def get_ob(msg):
#     ob = json.loads(msg)
#     goal_matrix = np.zeros(ARENA_WIDTH * ARENA_BREADTH).reshape(ARENA_WIDTH, ARENA_BREADTH)
#     mob_matrix = np.zeros(ARENA_WIDTH * ARENA_BREADTH).reshape(ARENA_WIDTH, ARENA_BREADTH)
#     self_matrix = np.zeros(ARENA_WIDTH * ARENA_BREADTH).reshape(ARENA_WIDTH, ARENA_BREADTH)
#     if "entities" in ob:
#         for entity in ob[u'entities']:
#             if entity["name"]==AGENT_TYPE:
#                 agent_yaw=entity["yaw"]
#                 agent_x=int(round(entity["x"]))
#                 agent_z=int(round(entity["z"]))
#                 self_matrix[agent_x,agent_z] = agent_yaw
#             if entity["name"] == GOAL_TYPE:
#                 goal_matrix[int(round(entity["x"])),int(round(entity["z"]))] += 1
#             if entity["name"] == MOB_TYPE:
#                 mob_matrix[int(round(entity["x"])),int(round(entity["z"]))] += 1
#
#     goal_matrix = goal_matrix.flatten()
#     mob_matrix = mob_matrix.flatten()
#     self_matrix = self_matrix.flatten()
#     observation = np.hstack((np.hstack((goal_matrix,mob_matrix)),self_matrix))
#     return observation


# RL take action and get next observation and reward
def execute_step(agent_host,action,DamageTaken):
    s_ = np.zeros(n_features)
    reward = 0
    done = False

    # print(action)
    # action=action[0]/180

    action =action[0]
    print(action)

    # take action
    agent_host.sendCommand("turn " + str(action))

    # agent_host.sendCommand(str(action_space[action]))

    time.sleep(0.1)
    world_state = agent_host.getWorldState()

    # get next observation and reward
    if world_state.number_of_observations_since_last_state > 0:
        msg = world_state.observations[-1].text
        ob = json.loads(msg)

        # if "isAlive" in ob:
        #     if not ob[u'isAlive']:
        #         done = True
        s_ = get_ob(msg)
        # print(ob)

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

    return s_, reward, DamageTaken


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
                <ContinuousMovementCommands turnSpeedDegs="360"/>
                <DiscreteMovementCommands/>
                <AbsoluteMovementCommands/>
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
# Create a pool of Minecraft Mod clients.
# By default, mods will choose consecutive mission control ports, starting at 10000,
# so running four mods locally should produce the following pool by default (assuming nothing else
# is using these ports):
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
    num_reps = 50000

#####################  hyper parameters  ####################

LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000  # most 1000000
BATCH_SIZE = 32

###############################  DDPG  ####################################


class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        self.a = self._build_a(self.S,)
        q = self._build_c(self.S, self.a, )
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Critic')
        ema = tf.train.ExponentialMovingAverage(decay=1 - TAU)          # soft replacement

        def ema_getter(getter, name, *args, **kwargs):
            return ema.average(getter(name, *args, **kwargs))

        target_update = [ema.apply(a_params), ema.apply(c_params)]      # soft update operation
        a_ = self._build_a(self.S_, reuse=True, custom_getter=ema_getter)   # replaced target parameters
        q_ = self._build_c(self.S_, a_, reuse=True, custom_getter=ema_getter)

        a_loss = - tf.reduce_mean(q)  # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=a_params)

        with tf.control_dependencies(target_update):    # soft replacement happened at here
            q_target = self.R + GAMMA * q_
            td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
            self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=c_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def learn(self):
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, reuse=None, custom_getter=None):
        w_initializer, b_initializer = tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)
        trainable = True if reuse is None else False
        with tf.variable_scope('Actor', reuse=reuse, custom_getter=custom_getter):
            net = tf.layers.dense(s, n_features,
                                  activation=tf.nn.relu,
                                  kernel_initializer=w_initializer,
                                  bias_initializer=b_initializer,
                                  name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim,
                                activation=tf.nn.softmax,
                                kernel_initializer=w_initializer,
                                bias_initializer=b_initializer,
                                name='a', trainable=trainable)
            # print("a before multiply:", a)
            # print("a before multiply:", a[0])
            # print("a_bound before multiply:", self.a_bound)
            # return a
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, reuse=None, custom_getter=None):
        trainable = True if reuse is None else False
        with tf.variable_scope('Critic', reuse=reuse, custom_getter=custom_getter):
            n_l1 = n_features
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)

            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)
#########################################################################################################################

current_life = 0

RL = DDPG(a_dim=n_actions,s_dim=n_features,a_bound=a_bound)

var = 3  # control exploration

step = 0
reward_his =[]
discounted_ep_rs_list=[]

for iRepeat in range(num_reps):
    mission_xml = getMissionXML(MOB_TYPE + " Apocalypse #" + str(iRepeat))
    my_mission = MalmoPython.MissionSpec(mission_xml,validate)
    max_retries = 3
    for retry in range(max_retries):
        try:
            # Set up a recording
            my_mission_record = MalmoPython.MissionRecordSpec(recordingsDirectory + "//" + "Mission_" + str(iRepeat) + ".tgz")
            my_mission_record.recordRewards()
            # my_mission_record = MalmoPython.MissionRecordSpec(recordingsDirectory)

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

    DamageTaken = []
    while not world_state.has_mission_begun:
        time.sleep(0.1)
        world_state = agent_host.getWorldState()
    while not world_state.number_of_observations_since_last_state > 0:
        time.sleep(0.02)
        world_state = agent_host.getWorldState()
    msg = world_state.observations[-1].text
    ob = json.loads(msg)
    if "DamageTaken" in ob:
            DamageTaken.append(ob[u'DamageTaken'])

    agent_host.sendCommand("move 1")    # run!

    # main loop:
    total_reward = 0
    total_commands = 0
    flash = False
    done=False
    observation = np.zeros(n_features)+0.1
    iteration_count =0


    while world_state.is_mission_running:
        world_state = agent_host.getWorldState()
        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text
            ob = json.loads(msg)
            observation=get_ob(msg)
            # if "Yaw" in ob:
            #     current_yaw = ob[u'Yaw']
            if "Life" in ob:
                life = ob[u'Life']
                if life < current_life:
                    # agent_host.sendCommand("chat aaaaaaaaargh!!!")
                    flash = True
                current_life = life
            if "entities" in ob:
                entities = [EntityInfo(**k) for k in ob["entities"]]
                drawMobs(entities, flash)

                # RL choose action based on observation
                a = RL.choose_action(observation)
                # print("action before:", a)

                action = np.clip(np.random.normal(a, var), -1, 1)  # add randomness to action selection for exploration

                # print("action:", action)

                # RL take action and get next observation and reward
                observation_, reward,DamageTaken = execute_step(agent_host,action,DamageTaken)
                # print(observation)

                # print(reward)
                # print(DamageTaken)
                if observation_[0]!=0.0 or observation_[1]!=0.0 or observation_[2]!=0.0 or observation_[3]!=0.0 or observation_[4]!=0.0:
                    RL.store_transition(observation, action, reward,observation_)
                    # print(observation_)

                # print("pointer:",RL.pointer)

                if RL.pointer > MEMORY_CAPACITY:
                    # print("start learn!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    var *= .99995  # decay the action randomness
                    RL.learn()

                total_commands += 1
                total_reward += reward

                # swap observation
                observation = observation_

        # break while loop when end of this episode
        # if done:
        #     print("done")
        #     break

        time.sleep(0.02)
        flash = False

    # discounted_ep_rs_list.append(RL.learn())
    # print(iRepeat)
    # if iRepeat % 3 == 0:
    #     plt.close('all')
    #     plt.plot(np.arange(len(np.hstack(discounted_ep_rs_list))),np.hstack(discounted_ep_rs_list))  # plot the episode vt
    #     plt.xlabel('episode steps')
    #     plt.ylabel('normalized state-action value')
    #     plt.show(block=False)
    # mission has ended.

    for error in world_state.errors:
        print("Error:",error.text)
    # if world_state.number_of_rewards_since_last_state > 0:
    #     # A reward signal has come in - see what it is:
    #     total_reward += world_state.rewards[-1].getValue()

    step += 1
    if step % 20 == 0:
        # print("pointer:", RL.pointer)
        plt.close('all')
        plt.plot(np.arange(len(reward_his)), reward_his)
        plt.ylabel('Reward')
        plt.xlabel('Training Steps')
        plt.show(block=False)

    reward_his.append(total_reward)
    print("game:",step,"pointer:",RL.pointer)
    print("We stayed alive for " + str(total_commands) + " commands, and scored " + str(total_reward)+" var "+str(var))
    print("Time used:",(time.clock() - start))
    time.sleep(1) # Give the mod a little time to prepare for the next mission.