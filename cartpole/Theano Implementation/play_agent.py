import gym
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import pickle
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras.layers.advanced_activations import LeakyReLU

env = gym.make('CartPole-v0')


def epsilon_greedy(Q, eps=0.1):
    "Compute the policy according to an epsilon greedy schedule"
    n = len(Q)
    eps2 = eps / (n - 1)
    pi = eps2 * np.ones(n)
    max_elem = np.argmax(Q)
    pi[max_elem] = 1 - eps
    action = int(np.random.choice(np.arange(len(Q)), 1, p=list(pi)))
    return pi, action


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

# =======================================================================================

with open('objs.pickle') as f:
    agent, performance = pickle.load(f)

# Uncomment the below lines if you want a video recording
# outdir = '/tmp/CartPole-Qnetwork'
# env.monitor.start(outdir, force=True)

for _ in range(3):
    s = env.reset()
    time = 0
    done = 0

    while (done != True and time < 200):
        env.render()
        Q = np.array(agent.net.predict(s.reshape(1,-1)))
        pi, a = epsilon_greedy(Q.ravel(),eps=0)
        sp, r, done, info = env.step(a)
        s = sp
        time += 1

    print "Resetting Environment. Balanced for ", time, " time steps (max possible = 200)"

# env.monitor.close()