import sys
sys.setrecursionlimit(5000)
import numpy as np
np.random.seed(10)
import time as t
import gym
import pylab as pl
import matplotlib.pyplot as plt
import keras.backend as K
from keras.layers import Input, Dense, Merge, merge
from keras.models import Model, Sequential, model_from_json
from keras.optimizers import SGD, Adam, RMSprop
from utils import *


# Default global values
LEARN_RATE = 1e-3
TAU = 1e-3
BATCH_SIZE = 16
GAMMA = 0.95

A_UNITS = [12, 12]
A_ACT = ['tanh', 'tanh', 'tanh']
C_UNITS = [12, 12]
C_ACT = ['relu', 'tanh', 'linear']


class LearnerNetwork:

    def __init__(self, state_dim, action_dim):
        ''' Create the Learner Networks | Three callable structures = net1, net2, actor
            net1 --> holds actor fixed and trains critic to match target values
            net2 --> holds critic fixed and trains actor to produce action that maximizes the Q value
            best_net --> for just keeping track of best_net so far
            actor --> can be used to predict the action given state
        '''

        # Make placeholders
        self.t_state = Input(shape=(state_dim,))
        self.t_exploration = Input(shape=(action_dim,))

        a1 = Dense(A_UNITS[0], activation=A_ACT[0])(self.t_state)
        a2 = Dense(A_UNITS[1], activation=A_ACT[1])(a1)
        a3 = Dense(output_dim=action_dim, activation=A_ACT[2])(a2)
        m1 = merge([self.t_exploration, a3], mode='sum')
        m2 = merge([self.t_state, m1], mode='concat')
        c1 = Dense(C_UNITS[0], activation=C_ACT[0])(m2)
        c2 = Dense(C_UNITS[1], activation=C_ACT[1])(c1)
        c3 = Dense(output_dim=1, activation=C_ACT[2])(c2)

        self.actor = Model(input=[self.t_state], output=[a3])
        self.net1  = Model(input=[self.t_state, self.t_exploration], output=[c3])
        self.net2  = Model(input=[self.t_state, self.t_exploration], output=[c3])

        # Compile networks
        opt_set1 = RMSprop(lr=LEARN_RATE)
        self.net1.compile(optimizer=opt_set1, loss='mse')
        opt_set2 = SGD(lr=LEARN_RATE)
        self.net2.compile(optimizer=opt_set2, loss='nmo')

        # Collect layers of actor and critic
        self.alayers = [1,2,3]
        self.clayers = [7,8,9]



class BestTrackingNetwork:

    def __init__(self, state_dim, action_dim):
        '''
            Creates a network which can be used to keep track of the best learner networks
            Make sure that the network structure is the same. Easiest way is to copy and paste
            content from the LearnerNetwork in here whenever changes are made
        '''

        # Make placeholders
        self.t_state = Input(shape=(state_dim,))
        self.t_exploration = Input(shape=(action_dim,))

        a1 = Dense(A_UNITS[0], activation=A_ACT[0])(self.t_state)
        a2 = Dense(A_UNITS[1], activation=A_ACT[1])(a1)
        a3 = Dense(output_dim=action_dim, activation=A_ACT[2])(a2)
        m1 = merge([self.t_exploration, a3], mode='sum')
        m2 = merge([self.t_state, m1], mode='concat')
        c1 = Dense(C_UNITS[0], activation=C_ACT[0])(m2)
        c2 = Dense(C_UNITS[1], activation=C_ACT[1])(c1)
        c3 = Dense(output_dim=1, activation=C_ACT[2])(c2)

        self.actor = Model(input=[self.t_state], output=[a3])
        self.net   = Model(input=[self.t_state, self.t_exploration], output=[c3])



class TargetNetwork:

    def __init__(self, state_dim, action_dim):
        ''' Create the target networks | Two callable structures: actor and critic '''

        # Make placeholders
        self.t_state = Input(shape=(state_dim,))
        self.t_exploration = Input(shape=(action_dim,))

        a1 = Dense(A_UNITS[0], activation=A_ACT[0])(self.t_state)
        a2 = Dense(A_UNITS[1], activation=A_ACT[1])(a1)
        a3 = Dense(output_dim=action_dim, activation=A_ACT[2])(a2)
        m1 = merge([self.t_exploration, a3], mode='sum')
        m2 = merge([self.t_state, m1], mode='concat')
        c1 = Dense(C_UNITS[0], activation=C_ACT[0])(m2)
        c2 = Dense(C_UNITS[1], activation=C_ACT[1])(c1)
        c3 = Dense(output_dim=1, activation=C_ACT[2])(c2)

        self.actor = Model(input=[self.t_state], output=[a3])
        self.critic = Model(input=[self.t_state, self.t_exploration], output=[c3])



class ActorCritic:

    def __init__(self, state_dim, action_dim, er_size=50000):

        # Initialize variables
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Get networks
        self.learner = LearnerNetwork(state_dim, action_dim)
        self.target  = TargetNetwork(state_dim, action_dim)
        self.best    = BestTrackingNetwork(state_dim, action_dim)
        self.target.critic.set_weights(self.learner.net1.get_weights())
        self.best.net.set_weights(self.learner.net1.get_weights())

        # Set experience replay
        self.er_s = []
        self.er_a = []
        self.er_r = []
        self.er_done = []
        self.er_sp = []

        self.er_size = er_size    # total size of mb, implement as queue
        self.whead = 0          # write head

        # Extract parameters once to avoid redundant computation
        self.learner_params = get_trainable_weights(self.learner.net1)
        self.target_params  = get_trainable_weights(self.target.critic)


    def update_networks(self, epochs=1, mb_size=BATCH_SIZE):

        mb_size = min(len(self.er_s), mb_size)
        Xstate, Xaction, critic_target = self.get_training_data(mb_size)

        # Define inputs for update routines
        exploration = Xaction - self.learner.actor.predict(Xstate, batch_size=mb_size)
        no_exploration = np.zeros((mb_size, self.action_dim), dtype=np.float32)

        # Update networks
        self.set_learn_mode('critic')
        self.learner.net1.fit([Xstate, exploration], critic_target, batch_size=mb_size, nb_epoch=epochs, verbose=0)
        self.set_learn_mode('actor')
        self.learner.net2.fit([Xstate, no_exploration], critic_target, batch_size=mb_size, nb_epoch=epochs, verbose=0)


    def get_training_data(self, mb_size=BATCH_SIZE):

        mini_batch = list(np.random.randint(len(self.er_s), size=mb_size))
        Xstate  = np.asarray([self.er_s[i] for i in mini_batch]).reshape(mb_size, self.state_dim)
        Xaction = np.array([self.er_a[i] for i in mini_batch]).reshape(mb_size, self.action_dim)
        Xsp     = np.asarray([self.er_sp[i] for i in mini_batch]).reshape(mb_size, self.state_dim)

        critic_target = np.random.rand(mb_size)
        no_exploration = np.zeros((mb_size, self.action_dim), dtype=np.float32).reshape(mb_size, -1)
        Q_pred = self.target.critic.predict([Xsp, no_exploration])  # predictions for Q(sp)

        for j, i in enumerate(mini_batch):
            if (self.er_done[i] == True):
                critic_target[j] = self.er_r[i]
            else:
                critic_target[j] = self.er_r[i] + GAMMA*Q_pred[j]

        critic_target = critic_target.reshape(mb_size, 1)

        return Xstate, Xaction, critic_target


    def update_target_networks(self, tau=TAU):
        weights = unpack_theta(self.learner.net1, self.learner_params)
        target_weights = unpack_theta(self.target.critic, self.target_params)
        target_weights = tau*weights + (1-tau)*target_weights
        set_model_params(self.target.critic, target_weights)


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


    def set_learn_mode(self, mode='critic'):
        if mode == 'critic':
            for l in self.learner.alayers:
                self.learner.net1.layers[l].trainable = False
            for l in self.learner.clayers:
                self.learner.net1.layers[l].trainable = True

        if mode == 'actor':
            for l in self.learner.alayers:
                self.learner.net2.layers[l].trainable = True
            for l in self.learner.clayers:
                self.learner.net2.layers[l].trainable = False


    def learn_policy(self,
                     env,
                     num_episodes=10,
                     learner_update_freq=1,
                     target_update_freq=1,
                     print_freq=10,
                     visual_freq=1e8,
                     save_freq=100,
                     mb_size=32,
                     epochs=1,
                     num_evals=20):

        max_steps = min(env.spec.timestep_limit, 250)
        performance = -5000*np.ones(num_episodes)
        best_ep = 0
        self.best.net.set_weights(self.learner.net1.get_weights())

        print "******** Starting learning process *************"

        for ep in range(num_episodes):
            done_flag = 0
            state = env.reset()
            time = 0

            while (done_flag!=True and time <= max_steps):
                actor_out = self.learner.actor.predict(state.reshape(1, -1))[0]
                action = actor_out + get_exploration(env, ep, num_episodes, high=0.5, low=0.3) # need to add exploration here
                next_state, reward, done_flag, _ = env.step(action)
                self.append_memory(state, action, reward, next_state, done_flag)
                state = next_state

                if (time % learner_update_freq == 0):
                    self.update_networks(epochs, mb_size)
                if (time % target_update_freq == 0):
                    self.update_target_networks(tau=TAU)

                time += 1

            performance[ep] = policy_evaluation(self.learner.actor, env, num_evals, max_steps=max_steps)

            if (ep % print_freq == 0):
                print "Now in episode: ", ep, " of ", num_episodes
                print "Agent best performance = ", performance[best_ep], " | current performance = ", np.mean(performance[0:ep])

            if (performance[ep] > performance[best_ep]):
                self.best.net.set_weights(self.learner.net1.get_weights())
                best_ep = ep
                print "Best agent switched! New performance = ", performance[best_ep]

            if (ep % save_freq == 0):
                self.best.net.save_weights('best_wts.h5', overwrite=True)

        print "**** Training Completed ****"
        print "Final Weights stored in : final_weights.h5"
        print "Performance of best agent = ", policy_evaluation(self.best.actor, env, 2 * num_evals, max_steps=max_steps)
        self.best.net.save_weights('final_weights.h5', overwrite=True)
        self.learner.net1.load_weights('final_weights.h5')
        print "Performance from file = ", policy_evaluation(self.learner.actor, env, 2 * num_evals, max_steps=max_steps)
        wts = self.best.net.get_weights()

        return wts, best_ep, performance



    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

