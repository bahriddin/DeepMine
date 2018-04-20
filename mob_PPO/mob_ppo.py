

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
# n_actions = len(action_space)
# n_actions=1
# a_bound=np.array([1.])

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
def step(agent_host,a,DamageTaken):
    s_ = np.zeros(n_features)
    reward = -3
    done = False

    # take action
    agent_host.sendCommand("turn " + str(a[0]/2))

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
        else:
            done=True
    else:
        done=True

    return s_, reward,done,DamageTaken


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
    num_reps = 100


###########################################################################################################################
"""
A simple version of Proximal Policy Optimization (PPO) using single thread.

Based on:
1. Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
2. Proximal Policy Optimization Algorithms (OpenAI): [https://arxiv.org/abs/1707.06347]

View more on my tutorial website: https://morvanzhou.github.io/tutorials

Dependencies:
tensorflow r1.2
gym 0.9.2
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import gym

EP_MAX = 1000
EP_LEN = 200
GAMMA = 0.9
A_LR = 0.0001
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = n_features, 1
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization


class PPO(object):

    def __init__(self):
        self.sess = tf.Session()
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')

        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, n_features, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = pi.prob(self.tfa) / oldpi.prob(self.tfa)
                surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4*METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # sometimes explode, this clipping is my solution
        else:   # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            l1 = tf.layers.dense(self.tfs, n_features, tf.nn.relu, trainable=trainable)
            mu = 2 * tf.layers.dense(l1, A_DIM, tf.nn.tanh, trainable=trainable)
            sigma = tf.layers.dense(l1, A_DIM, tf.nn.softplus, trainable=trainable)
            norm_dist = tf.distributions.Normal(loc=mu, scale=sigma)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return norm_dist, params

    def choose_action(self, s):
        s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})[0]
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})[0, 0]


########################################################################################################################
current_life = 0

ppo = PPO()
all_ep_r = []

# step = 0
reward_his =[]

for ep in range(num_reps):
    mission_xml = getMissionXML(MOB_TYPE + " Apocalypse #" + str(ep))
    my_mission = MalmoPython.MissionSpec(mission_xml,validate)
    max_retries = 3
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

    agent_host.sendCommand("move 1")    # run!

    # while not world_state.number_of_observations_since_last_state > 0:
    #     time.sleep(0.02)
    #     world_state = agent_host.getWorldState()

    # main loop:
    total_reward = 0
    total_commands = 0
    flash = False
    # done=False
    # s = np.zeros(n_features)

    buffer_s, buffer_a, buffer_r = [], [], []
    ep_r = 0

    # print(world_state.is_mission_running)
    msg = world_state.observations[-1].text
    ob = json.loads(msg)
    s = get_ob(msg)
    # print("initial s:",s)

    DamageTaken = []
    if "DamageTaken" in ob:
        DamageTaken.append(ob[u'DamageTaken'])
    t=0

    while world_state.is_mission_running:  # in one episode
        world_state = agent_host.getWorldState()
        if world_state.number_of_observations_since_last_state > 0:
            msg = world_state.observations[-1].text
            s = get_ob(msg)
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

                # print("s:",s)
                # RL choose action based on observation
                a = ppo.choose_action(s)
                # a=a/2

                # take action and get next observation and reward
                s_, r,done, DamageTaken = step(agent_host, a, DamageTaken)

                print("step:",t,"action :", a,"reward:",r)
                # print("s_:",s_)

                buffer_s.append(s)
                buffer_a.append(a)
                buffer_r.append((r + 20) / 20)  # normalize reward, find to be useful

                s = s_
                ep_r += r

                # update ppo
                if (t + 1) % BATCH == 0 or done:
                    print("update")
                    v_s_ = ppo.get_v(s_)
                    discounted_r = []
                    for r in buffer_r[::-1]:
                        v_s_ = r + GAMMA * v_s_
                        discounted_r.append(v_s_)
                    discounted_r.reverse()

                    bs, ba, br = np.vstack(buffer_s), np.vstack(buffer_a), np.array(discounted_r)[:, np.newaxis]
                    buffer_s, buffer_a, buffer_r = [], [], []
                    ppo.update(bs, ba, br)

                t+=1

                # break while loop when end of this episode
                if done:
                    print("done")
                    break

        time.sleep(0.02)
        flash = False

    if ep == 0: all_ep_r.append(ep_r)
    else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
    print(
        'Ep: %i' % ep,
        "|Ep_r: %i" % ep_r,
        ("|We stayed alive for " + str(total_commands) + " commands"),
        ("|Time used:", (time.clock() - start)),
        ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
    )

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

    # step += 1
    # if step % 20 == 0:
    #     # print("pointer:", RL.pointer)
    #     plt.close('all')
    #     plt.plot(np.arange(len(reward_his)), reward_his)
    #     plt.ylabel('Reward')
    #     plt.xlabel('Training Steps')
    #     plt.show(block=False)

    time.sleep(1) # Give the mod a little time to prepare for the next mission.

plt.plot(np.arange(len(all_ep_r)), all_ep_r)
plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()