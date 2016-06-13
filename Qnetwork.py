'''
    Reinforcement learning agent based on Q network
    Currently implimenting for discrete action spaces
    For continuous control, need to do actor critic
    Aravind Rajeswaran, 13th June 2016
'''
import warnings
warnings.filterwarnings("ignore")


import gym
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

env = gym.make('CartPole-v0')

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    sf = np.exp(x)
    sf = sf/np.sum(sf, axis=0)
    return sf

class QN(object):

    def __init__(self, num_inputs, num_outputs):
        self.nx = num_inputs
        self.ny = num_outputs     
        self.net = MLPRegressor(hidden_layer_sizes=(10,10),
                               max_iter=1,
                               algorithm='sgd',
                               learning_rate='adaptive',
                               warm_start=True,
                               momentum=0.9,
                               nesterovs_momentum=True
                               )
        
        self.initialize_network()

        # set experience replay
        self.mbsize = 5     # mini-batch size
        self.er_s =    []
        self.er_a =    []
        self.er_r =    []
        self.er_done = []
        self.er_sp =   []

    def initialize_network(self):
        # function to initialize network weights
        xtrain = np.random.rand(100, self.nx)
        ytrain = np.random.rand(100, self.ny)
        self.net.fit(xtrain, ytrain)
        
    def update_network(self):
        # function updates network by sampling a mini-batch from the ER
        # Prepare train data
        chosen = list( np.random.randint(len(self.er_s), size=min(len(self.er_s),self.mbsize)) )
        Xtrain = np.asarray( [self.er_s[i] for i in chosen] )
        # calculate target
        target = np.random.rand(len(chosen), self.ny)

        #print target

        for j,i in enumerate(chosen):
            # do a forward pass through s_p
            Q_p = self.net.predict(self.er_sp[i])
            #print "QP : \n", Q_p
            target[j,:] = Q_p
            #print "target[j,:] : \n", target[j,:], type(target[j,:])

            if(self.er_done[i] == True):
                target[j,self.er_a[i]] = self.er_r[i]
            else:
                target[j,self.er_a[i]] = self.er_r[i] + 0.99*max(max(Q_p))   # Q_p is list of list (why?)

        # fit the network
        self.net.fit(Xtrain, target) # single step of SGD
        
    def append_memory(self,s,a,r,sp,done):
        self.er_s.append(s)
        self.er_a.append(a)
        self.er_r.append(r)
        self.er_sp.append(sp)
        self.er_done.append(done)

# Set up the Q learning agent
agent = QN(4,2)
state = env.reset()

# Initial data build up
done_flag=0
for i in range(500):
    # Reset env once in a while
    if(done_flag==True or i%100==0 and i!=0):
        state = env.reset()

    action = env.action_space.sample()
    next_state, reward, done_flag, info = env.step(action)
    if(done_flag):
        reward=-10

    agent.append_memory(state,action,reward,next_state,done_flag)
    state = next_state

print "Initial memory built!!"

print "******** Starting learning process *************"
num_plays = 100000
episode_length = 500
update_freq = 50

performance = np.random.rand(num_plays)

state = env.reset()
for i in range(num_plays):
    if(done_flag==True or i%episode_length==0 and i!=0):
        state = env.reset()
        done_flag=0

    Q_pred = np.array(agent.net.predict(state))
    pi = softmax(Q_pred.ravel())
    action = np.random.choice(np.array([0,1]),1,p=list(pi))
    action = int(action)
    next_state, reward, done_flag, info = env.step(action)
    if(done_flag==True):
        reward = -10

    agent.append_memory(state,action,reward,next_state,done_flag)
    state = next_state
    performance[i] = reward

    if(i%update_freq==0):
        agent.update_network()
        print "Network updated...."

plt.plot(performance[-500:])
plt.show()

# Evaluate performance
print "Evaluating performance............."
for _ in range(2):
    state = env.reset()
    for t in range(50):
        env.render()

        Q_pred = np.array(agent.net.predict(state))
        pi = softmax(Q_pred.ravel())
        action = np.random.choice(np.array([0, 1]), 1, p=list(pi))
        action = int(action)

        next_state, reward, done_flag, info = env.step(action)

        if done_flag:
            break


