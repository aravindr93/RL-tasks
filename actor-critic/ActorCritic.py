import numpy as np
import keras.backend as K
from keras.layers import Input, Dense, Merge, merge
from keras.models import Model, Sequential
from utils import *
from derivative_routines import *

LEARN_RATE = 1e-4
TAU = 0.001
BATCH_SIZE = 64
GAMMA = 0.99


class LearnerNetwork:

    def __init__(self, state_dim, action_dim):
        ''' Create the Learner Networks | Three callable structures = net1, net2, actor
            net1 --> holds actor fixed and trains critic to match target values
            net2 --> holds critic fixed and trains actor to produce action that maximizes the Q value
            actor --> can be used to predict the action given state
        '''

        # Make placeholders
        self.t_state = Input(shape=(state_dim,))
        self.t_exploration = Input(shape=(action_dim,))
        self.t_target = Input(shape=(1,))

        # 1) Network that updates the critic part only (call: net1)
        net1_ad1 = Dense(5, activation='relu')(self.t_state)
        net1_ad2 = Dense(5, activation='relu')(net1_ad1)
        net1_ad3 = Dense(output_dim=action_dim, activation='linear')(net1_ad2)
        # Merge layers
        net1_m1 = merge([self.t_exploration, net1_ad3], mode='sum')
        net1_m2 = merge([self.t_state, net1_m1], mode='concat')
        # Critic layers
        net1_cd1 = Dense(5, activation='relu')(net1_m2)
        net1_cd2 = Dense(5, activation='relu')(net1_cd1)
        net1_cd3 = Dense(output_dim=1, activation='linear')(net1_cd2)
        # setup net1
        self.net1 = Model(input=[self.t_state, self.t_exploration], output=[net1_cd3], name='net1')
        # Freeze actor layers
        self.net1.layers[1].trainable = False  # actor_dense_1
        self.net1.layers[2].trainable = False  # actor_dense_2
        self.net1.layers[3].trainable = False  # actor_dense_3
        # compile network with mse loss (wrt target)
        self.net1.compile(optimizer='rmsprop', loss='mse')

        # 2) Network that updates the actor part only (call: net2)
        net2_ad1 = Dense(5, activation='relu')(self.t_state)
        net2_ad2 = Dense(5, activation='relu')(net2_ad1)
        net2_ad3 = Dense(output_dim=action_dim, activation='linear')(net2_ad2)
        self.actor = Model(input=[self.t_state], output=[net2_ad3], name='actor')  # useful for getting predictions
        # Merge layers
        net2_m1 = merge([self.t_exploration, net2_ad3], mode='sum')
        net2_m2 = merge([self.t_state, net2_m1], mode='concat')
        # Critic layers
        net2_cd1 = Dense(5, activation='relu')(net2_m2)
        net2_cd2 = Dense(5, activation='relu')(net2_cd1)
        net2_cd3 = Dense(output_dim=1, activation='linear')(net2_cd2)
        # setup net2
        self.net2 = Model(input=[self.t_state, self.t_exploration], output=[net2_cd3], name='net2')
        # Freeze critic layers
        self.net2.layers[7].trainable = False
        self.net2.layers[8].trainable = False
        self.net2.layers[9].trainabel = False
        # compile network with nmo loss (negative mean output)
        self.net2.compile(optimizer='rmsprop', loss='nmo')

        # Make the weights match!
        self.net2.set_weights(self.net1.get_weights())

    def sync_net1(self):
        ''' Update net1 to match net2 '''
        self.net1.set_weights(self.net2.get_weights())

    def sync_net2(self):
        ''' Update net2 to match net1 '''
        self.net2.set_weights(self.net1.get_weights())

class TargetNetwork:

    def __init__(self, state_dim, action_dim):
        ''' Create the target networks | Two callable structures: actor and critic '''

        # Make placeholders
        self.t_state = Input(shape=(state_dim,))
        self.t_exploration = Input(shape=(action_dim,))
        self.t_target = Input(shape=(1,))

        ad1 = Dense(5, activation='relu')(self.t_state)
        ad2 = Dense(5, activation='relu')(ad1)
        ad3 = Dense(output_dim=action_dim, activation='linear')(ad2)
        m1 = merge([self.t_exploration, ad3], mode='sum')
        m2 = merge([self.t_state, m1], mode='concat')
        cd1 = Dense(5, activation='relu')(m2)
        cd2 = Dense(5, activation='relu')(cd1)
        cd3 = Dense(output_dim=1, activation='linear')(cd2)

        self.actor = Model(input=[self.t_state], output=[ad3])
        self.critic = Model(input=[self.t_state, self.t_exploration], output=[cd3])


class ActorCritic:

    def __init__(self, state_dim, action_dim):

        # Initialize variables
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Define training network
        self.learner = LearnerNetwork(state_dim, action_dim)

        # Define target network
        self.target = TargetNetwork(state_dim, action_dim)

        # Copy weights to make the two sets match
        self.target.critic.set_weights(self.learner.net1.get_weights())

        # Set experience replay
        self.er_s = []
        self.er_a = []
        self.er_r = []
        self.er_done = []
        self.er_sp = []

        self.er_size = 2000  # total size of mb, implement as queue
        self.whead = 0  # write head


    def update_networks(self, epochs=1):

        mb_size = min(len(self.er_s), BATCH_SIZE)
        Xtrain_state, Xtrain_action, critic_target = self.get_training_data(mb_size)

        # Define inputs for update routines
        exploration = Xtrain_action - self.learner.actor.predict(Xtrain_state, batch_size=mb_size)
        no_exploration = np.zeros((mb_size, self.action_dim), dtype=np.float32)

        # Hold actor fixed and update critic
        self.learner.net1.fit([Xtrain_state, exploration], [critic_target], nb_epoch=epochs, verbose=0)
        self.learner.sync_net2()  # makes net2 match net1
        # Hold critic fixed and update actor
        self.learner.net2.fit([Xtrain_state, no_exploration], [critic_target], nb_epoch=1, verbose=0)
        self.learner.sync_net1()  # makes net1 match net2


    def get_training_data(self, mb_size=BATCH_SIZE):

        mini_batch = list(np.random.randint(len(self.er_s), size=mb_size))
        Xtrain_state = np.asarray([self.er_s[i] for i in mini_batch]).reshape(mb_size, self.state_dim)
        Xtrain_action = np.array([self.er_a[i] for i in mini_batch]).reshape(mb_size, self.action_dim)
        critic_target = np.random.rand(mb_size)

        no_exploration = np.zeros((1, self.action_dim), dtype=np.float32).reshape(1, -1)

        for j, i in enumerate(mini_batch):
            if (self.er_done[i] == True):
                critic_target[j] = self.er_r[i]
            else:
                critic_target[j] = self.er_r[i] + \
                                   GAMMA * self.target.critic.predict([self.er_sp[i].reshape(1, -1), no_exploration])

        critic_target = critic_target.reshape(mb_size, 1)

        return Xtrain_state, Xtrain_action, critic_target


    def update_target_networks(self, tau=TAU):
        weights = unpack_theta(self.learner.net1)
        target_weights = unpack_theta(self.target.critic)
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


    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

