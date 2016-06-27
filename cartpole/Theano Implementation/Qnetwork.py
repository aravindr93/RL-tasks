'''
    Theano Implimentation
    Reinforcement learning agent based on Q network
    Currently implimenting for discrete action spaces
    For continuous control, need to do actor critic
    Aravind Rajeswaran, 17th June 2016
'''
import sys
sys.setrecursionlimit(2000)
import numpy as np
np.random.seed(10)
import time as t
import gym
import pylab as pl
import matplotlib.pyplot as plt
import pickle
import copy
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.layers.advanced_activations import LeakyReLU


env = gym.make('CartPole-v0')


def softmax(x):
    "Compute softmax values for each sets of scores in x."
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


def policy_evaluation(agent, env, num_episodes=10, max_steps=250):
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
    "Q-network class object"
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.create_network()

        # set experience replay
        self.mbsize = 128       # mini-batch size
        self.er_s = []
        self.er_a = []
        self.er_r = []
        self.er_done = []
        self.er_sp = []

        self.er_size = 2000     # total size of mb, implement as queue
        self.whead = 0          # write head

    def create_network(self):
        "function to create the network"
        self.net = Sequential()
        self.net.add(Dense(50, input_dim=self.nx, init='uniform',
                      activation='relu'))
        self.net.add(Dense(10, init='uniform', activation='relu'))
        self.net.add(Dense(output_dim=self.ny, init='uniform'))

        # opt_settings = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        opt_settings = Adam(lr=0.001)
        self.net.compile(loss="mean_squared_error", optimizer=opt_settings)

    def update_network(self):
        "function updates network by sampling a mini-batch from the ER"
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
        self.net.fit(Xtrain, target, nb_epoch=1, batch_size=self.mbsize, verbose=0)  # single step of SGD

    def append_memory(self, s, a, r, sp, done):
        "append memory of agent with the given samples; implemented as queue"
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

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

start_time = t.time()
# Set up the Q learning agent
agent = QN(4,2)
state = env.reset()

# =================================================================================
# Initial data build up
done_flag = 0
for i in range(500):

    if (done_flag == True):
        state = env.reset()

    action = env.action_space.sample()
    next_state, reward, done_flag, info = env.step(action)
    agent.append_memory(state, action, reward, next_state, done_flag)
    state = next_state

print "Initial memory built!!"

# Initial Training for a few steps
for _ in range(5):
    agent.update_network()

print "Initial network performance = ", policy_evaluation(agent, env, 5)
# =================================================================================

print "******** Starting learning process *************"
num_episodes = 250
update_freq = 1      # update after how many steps (within each episode)
print_freq = 1	     # how often to print (episodes)

max_steps = 250      # number of steps before resetting env and beginning next episode

performance = np.zeros(num_episodes)
best_ep = 0
best_agent = copy.deepcopy(agent)

for ep in range(num_episodes):
    done_flag = 0
    state = env.reset()
    time = 0

    while (done_flag!=True and time<=max_steps):
        Q_pred = np.array(agent.net.predict(state.reshape(1, -1)))
        # pi, action = epsilon_greedy(Q_pred.ravel(),eps=0.25)
        pi, action = softmax_policy(Q_pred.ravel())
        next_state, reward, done_flag, info = env.step(action)

        agent.append_memory(state, action, reward, next_state, done_flag)
        state = next_state

        if (time % update_freq == 0):
            agent.update_network()

        time += 1

    performance[ep] = policy_evaluation(agent, env, 50, max_steps=max_steps)

    if (ep % print_freq == 0):
        print "Now in episode: ", ep, " of ", num_episodes
        print "Agent performance = ", performance[ep]

    if (performance[ep] > performance[best_ep]):
        best_agent = copy.deepcopy(agent)
        best_ep = ep

end_time = t.time()
print "Total time", (end_time - start_time)
plt.plot(performance[-100:])
plt.show()

# Save agent to file
with open('acrobot.pickle', 'wb') as f:
    pickle.dump([best_agent, performance], f)

# Evaluate performance of best agent
print "Performance of best agent = ", policy_evaluation(best_agent, env, 200, max_steps=max_steps)

# Visually inspect performance
agent = copy.deepcopy(best_agent)
for _ in range(10):
    s = env.reset()
    time = 0
    done = 0

    while (done != True and time < max_steps):
        env.render()
        Q = np.array(agent.net.predict(s.reshape(1,-1)))
        pi, a = epsilon_greedy(Q.ravel(),eps=0)
        sp, r, done, info = env.step(a)
        s = sp
        time += 1
