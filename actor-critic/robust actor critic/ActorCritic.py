import numpy as np
import keras.backend as K
import keras
from keras.layers import Input, Dense, Merge, merge
from keras.models import Model, Sequential
from utils import *
from derivative_routines import *

LEARN_RATE = 0.001
TAU = 0.001
BATCH_SIZE = 64
GAMMA = 0.99


class LearnerNetwork:

    def __init__(self, state_dim, action_dim, param_dim):
        ''' Create the Learner Networks | Three callable structures = net1, net2, actor
            net1 --> for training critic
            net2 --> for training actor
            actor --> can be used to predict the action given state
            alayers and clayers --> contain actor and critic layers; freeze appropriate layers b4 net.fit
        '''

        # Make placeholders
        self.t_state = Input(shape=(state_dim,))
        self.t_exploration = Input(shape=(action_dim,))
        self.t_param = Input(shape=(param_dim,))
        self.t_target = Input(shape=(1,))

        # Actor part of net
        a1 = Dense(10, activation='relu')(self.t_state)
        a2 = Dense(10, activation='relu')(a1)
        a3 = Dense(output_dim=action_dim, activation='linear')(a2)

        # Merge actor out with exploration
        m1 = merge([self.t_exploration, a3], mode='sum')
        m2 = merge([self.t_state, m1], mode='concat')

        # Pass (s,a) through a few layers
        c1 = Dense(10, activation='relu')(m2)
        c2 = Dense(5, activation='tanh')(c1)

        # Pass params through a few layers
        p1 = Dense(10, activation='relu')(self.t_param)
        p2 = Dense(5, activation='tanh')(p1)

        # Merge c2 and p2
        m3 = merge([c2, p2], mode='concat')

        # Critic layers
        c3 = Dense(10, activation='relu')(m3)
        c4 = Dense(5, activation='relu')(c3)
        c5 = Dense(output_dim=1, activation='linear')(c4)

        # Create the networks
        self.actor = Model(input=[self.t_state], output=[a3])
        self.net1 =  Model(input=[self.t_state, self.t_exploration, self.t_param], output=[c5])
        self.net2 =  Model(input=[self.t_state, self.t_exploration, self.t_param], output=[c5])

        # Compile networks
        sgd = keras.optimizers.SGD(lr=LEARN_RATE, momentum=0.9, decay=1e-6, nesterov=True)
        self.net1.compile(optimizer=sgd, loss='mse')
        self.net2.compile(optimizer=sgd, loss='nmo')

        # Collect layers of actor and critic
        self.alayers = [1, 2, 3]  # actor layers
        self.clayers = [8, 9, 10, 11, 13, 14, 15]


class TargetNetwork:

    def __init__(self, state_dim, action_dim, param_dim):
        ''' Create the target networks | Two callable structures: actor and critic '''

        # Make placeholders
        self.t_state = Input(shape=(state_dim,))
        self.t_exploration = Input(shape=(action_dim,))
        self.t_param = Input(shape=(param_dim,))
        self.t_target = Input(shape=(1,))

        a1 = Dense(10, activation='relu')(self.t_state)
        a2 = Dense(10, activation='relu')(a1)
        a3 = Dense(output_dim=action_dim, activation='linear')(a2)
        m1 = merge([self.t_exploration, a3], mode='sum')
        m2 = merge([self.t_state, m1], mode='concat')
        c1 = Dense(10, activation='relu')(m2)
        c2 = Dense(5, activation='tanh')(c1)
        p1 = Dense(10, activation='relu')(self.t_param)
        p2 = Dense(5, activation='tanh')(p1)
        m3 = merge([c2, p2], mode='concat')
        c3 = Dense(10, activation='relu')(m3)
        c4 = Dense(5, activation='relu')(c3)
        c5 = Dense(output_dim=1, activation='linear')(c4)

        self.actor  = Model(input=[self.t_state], output=[a3])
        self.critic = Model(input=[self.t_state, self.t_exploration, self.t_param], output=[c5])


class ActorCritic:

    def __init__(self, state_dim, action_dim, param_dim):
        # Initialize variables
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.param_dim = param_dim

        # Get networks
        self.learner = LearnerNetwork(state_dim, action_dim, param_dim)
        self.target  = TargetNetwork(state_dim, action_dim, param_dim)
        self.target.critic.set_weights(self.learner.net1.get_weights())

        # Set experience replay   (s,a,r,sp,p,done)
        self.er_s = []
        self.er_a = []
        self.er_r = []
        self.er_sp = []
        self.er_p = []
        self.er_done = []

        self.er_size = 5000  # total size of mb, implement as queue
        self.whead = 0  # write head

        # Extract parameters once to avoid redundant computation
        self.learner_params = get_trainable_weights(self.learner.net1)
        self.target_params  = get_trainable_weights(self.target.critic)


    def update_learners(self, epochs=1):

        mb_size = min(len(self.er_s), BATCH_SIZE)
        Xstate, Xaction, Xparams, critic_target = self.get_training_data(mb_size)

        # Define inputs for update routines
        exploration = Xaction - self.learner.actor.predict(Xstate, batch_size=mb_size)
        no_exploration = np.zeros((mb_size, self.action_dim), dtype=np.float32)

        # Update nets
        self.set_learn_mode('critic')
        self.learner.net1.fit([Xstate, exploration, Xparams], critic_target, batch_size=mb_size, nb_epoch=epochs, verbose=0)
        self.set_learn_mode('actor')
        self.learner.net2.fit([Xstate, no_exploration, Xparams], critic_target, batch_size=mb_size, nb_epoch=epochs, verbose=0)


    def get_training_data(self, mb_size=BATCH_SIZE):

        mini_batch = list(np.random.randint(len(self.er_s), size=mb_size))
        Xstate  = np.asarray([self.er_s[i] for i in mini_batch]).reshape(mb_size, self.state_dim)
        Xaction = np.asarray([self.er_a[i] for i in mini_batch]).reshape(mb_size, self.action_dim)
        Xparams = np.asarray([self.er_p[i] for i in mini_batch]).reshape(mb_size, self.param_dim)
        Xsp     = np.asarray([self.er_sp[i] for i in mini_batch]).reshape(mb_size, self.state_dim)
        critic_target = np.random.rand(mb_size)

        no_exploration = np.zeros((mb_size, self.action_dim), dtype=np.float32).reshape(mb_size, -1)

        Q_pred = self.target.critic.predict([Xsp, no_exploration, Xparams])  # this is the Q_pred from sp

        for j, i in enumerate(mini_batch):
            if self.er_done[i]:
                critic_target[j] = self.er_r[i]
            else:
                critic_target[j] = self.er_r[i] + GAMMA*Q_pred[j]

        critic_target = critic_target.reshape(mb_size, 1)

        return Xstate, Xaction, Xparams, critic_target


    def update_target_networks(self, tau=TAU):
        weights = unpack_theta(self.learner.net1, self.learner_params)
        target_weights = unpack_theta(self.target.critic, self.target_params)
        target_weights = tau*weights + (1-tau)*target_weights
        set_model_params(self.target.critic, target_weights)


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


    def append_memory(self, s, a, r, sp, p, done):
        if (len(self.er_s) < self.er_size):
            self.er_s.append(s)
            self.er_a.append(a)
            self.er_r.append(r)
            self.er_sp.append(sp)
            self.er_p.append(p)
            self.er_done.append(done)
            self.whead = (self.whead + 1) % self.er_size
        else:
            self.er_s[self.whead] = s
            self.er_a[self.whead] = a
            self.er_r[self.whead] = r
            self.er_sp[self.whead] = sp
            self.er_p[self.whead] = p
            self.er_done[self.whead] = done
            self.whead = (self.whead+1) % self.er_size


    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

