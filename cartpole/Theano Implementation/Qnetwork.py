"""
Theano implementation of Q learning.
"""

'''
    Reinforcement learning agent based on Q network
    Currently implimenting for discrete action spaces
    For continuous control, need to do actor critic
    Aravind Rajeswaran, 13th June 2016
'''
import time as t
import gym
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import pickle
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU
#from sklearn.neural_network import MLPRegressor

env = gym.make('CartPole-v0')


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    sf = np.exp(x)
    sf = sf / np.sum(sf, axis=0)
    return sf


def softmax_policy(Q):
    "Compute the softmax policy and choose action according to Q"
    pi = softmax(Q)
    action = int(np.random.choice(np.arange(len(Q)), 1, p=list(pi)))
    return pi, action


def epsilon_greedy(Q, eps=0.1):
    "Compute the policy according to an epsilon greedy schedule"
    n = len(Q)
    eps2 = eps / (n - 1)
    pi = eps2 * np.ones(n)
    max_elem = np.argmax(Q)
    pi[max_elem] = 1 - eps
    action = int(np.random.choice(np.arange(len(Q)), 1, p=list(pi)))
    return pi, action


def policy_evaluation(agent, env, num_episodes=10, max_steps=2000):
    "Evaluate quality of policy by performing rollouts"
    evaluation = 0.0
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        done = 0
        time = 0
        while (done != True and time < max_steps):
            Q = np.array(agent.net.predict(state.reshape(1, -1)))
            a = int(np.argmax(Q.ravel()))
            sp, r, done, info = env.step(a)
            if (done != True):
                episode_reward += r
            state = sp
            time += 1
        evaluation = evaluation + episode_reward/num_episodes

    env.reset()
    return evaluation


class QN(object):
    def __init__(self, env):
        self.nx = 4
        self.ny = 2
        self.env = env
        self.net = self.create_network(env.observation_space, \
                                        env.action_space)
        self.initialize_network()
        
        
        
        # set experience replay
        self.mbsize = 128 # mini-batch size
        self.er_s = []
        self.er_a = []
        self.er_r = []
        self.er_done = []
        self.er_sp = []

        self.er_size = 2000  # total size of mb, impliment as queue
        self.whead = 0  # write head

    def create_network(self, obs_space, ac_space):
        net = Sequential()
        net.add(Dense(50, input_dim=4, init='uniform',\
                            activation='relu'))
        net.add(Dense(10, init='uniform',\
                            activation='relu'))
        net.add(Dense(output_dim=2, init='uniform'))                            
        return net
    
    def initialize_network(self):
        # function to initialize network weights
        sgd = SGD(lr=0.001, momentum=0.9, nesterov=True)
        self.net.compile(loss="mean_squared_error", optimizer='sgd')

    def update_network(self):
        # function updates network by sampling a mini-batch from the ER
        # Prepare train data
        chosen = list(np.random.randint(len(self.er_s), size=min(len(self.er_s), self.mbsize)))
        Xtrain = np.asarray([self.er_s[i] for i in chosen])
        # calculate target
        target = np.random.rand(len(chosen), self.ny)

        for j, i in enumerate(chosen):
            # do a forward pass through s and sp
            Q_s = self.net.predict(self.er_s[i].reshape(1, -1))
            Q_sp = self.net.predict(self.er_sp[i].reshape(1, -1))
            target[j, :] = Q_s  # target initialized to current prediction

            if (self.er_done[i] == True):
                target[j, self.er_a[i]] = self.er_r[i]  # if end of episode, target is terminal reward
            else:
                target[j, self.er_a[i]] = self.er_r[i] + 0.9 * max(max(Q_sp))  # Q_sp is list of list (why?)

        # fit the network
        #print "hhhhhh"
        self.net.fit(Xtrain, target, nb_epoch=1, batch_size=256)  # single step of SGD
    
    def len_of_replay(self):
        return len(self.er_s)

    def append_memory(self, s, a, r, sp, done):
        if (len(self.er_s) < self.er_size):
            self.er_s.append(s)
            self.er_a.append(a)
            self.er_r.append(r)
            self.er_sp.append(sp)
            self.er_done.append(done)
            self.whead = (self.whead + 1) % self.er_size
        else:
            self.er_s[self.whead] = s
            self.er_a[self.whead] = a
            self.er_r[self.whead] = r
            self.er_sp[self.whead] = sp
            self.er_done[self.whead] = done
            self.whead = (self.whead+1) % self.er_size


start_time = t.time()
# Set up the Q learning agent
agent = QN(env)
state = env.reset()

# =================================================================================
# Initial data build up
done_flag = 0
for i in range(200):
    # Reset env once in a while
    if (done_flag == True):
        state = env.reset()

    action = env.action_space.sample()
    next_state, reward, done_flag, info = env.step(action)
    # if(done_flag):
    # reward=-10

    agent.append_memory(state, action, reward, next_state, done_flag)
    state = next_state

print "Initial memory built!!"


# Initial Training
for _ in range(2):
    agent.update_network()

#agent.update_network()
#print "Initial network loss = ", history.history
# =================================================================================

print "******** Starting learning process *************"
num_episodes = 500
update_freq = 1      # update after how many steps (within each episode)
print_freq = 20      # how often to print (episodes)

performance = np.zeros(num_episodes)
error_decay = np.zeros(num_episodes)

for ep in range(num_episodes):
    done_flag = 0
    state = env.reset()
    time = 0

    while (done_flag!=True and time<=2000):
        Q_pred = np.array(agent.net.predict(state.reshape(1, -1)))
        # pi, action = softmax_policy(Q_pred.ravel())
        pi, action = softmax_policy(Q_pred.ravel())
        next_state, reward, done_flag, info = env.step(action)

        agent.append_memory(state, action, reward, next_state, done_flag)
        state = next_state

        if (time % update_freq == 0):
            agent.update_network()

        time += 1
    
    performance[ep] = policy_evaluation(agent, env, 5)
    #error_decay[ep] = agent.net.loss_
    
    if (ep % print_freq == 0):
        print "Now in episode: ", ep, " of ", num_episodes
        print "Agent performance = ", performance[ep]
        print "Size of er: ", agent.len_of_replay()

end_time = t.time()
print "Total time", (end_time - start_time)
plt.plot(performance[-100:])
plt.show()


# Save agent to file
with open('objs.pickle', 'w') as f:
    pickle.dump(agent, f)

# Visually inspect performance
for _ in range(20):
    s = env.reset()
    time = 0
    done = 0

    while (done != True and time < 1500):
        env.render()
        Q = np.array(agent.net.predict(s.reshape(1,-1)))
        pi, a = epsilon_greedy(Q.ravel(),eps=0)
        sp, r, done, info = env.step(a)
        s = sp
