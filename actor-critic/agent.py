import gym
import theano
from utils import *
from actor_net import *
from critic_net import *
import tensorflow as tf
import numpy as np

TAU = 0.001
BATCH_SIZE = 64
GAMMA = 0.99
class ActorCriticAgent():
    
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.actor_net = Actor(state_dim, action_dim)
        self.critic_net = Critic(state_dim, action_dim)
        self.actor_prime_net = Actor(state_dim, action_dim)
        self.critic_prime_net = Critic(state_dim, action_dim)
        
        self.update_prime_nets(tau=1)
        # set experience replay
        self.er_s = []
        self.er_a = []
        self.er_r = []
        self.er_done = []
        self.er_sp = []

        self.er_size = 2000  # total size of mb, impliment as queue
        self.whead = 0       # write head
    
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
    
    #check this against the paper
    def update_network(self):
        mb_size = min(len(self.er_s), BATCH_SIZE)
        mini_batch = list(np.random.randint(len(self.er_s), size=mb_size))
        Xtrain_state =  np.asarray([self.er_s[i] for i in mini_batch])
        Xtrain_action = np.array([self.er_a[i] for i in mini_batch])
        critic_target = np.random.rand(mb_size)

        for j, i in enumerate(mini_batch):
            action_sp = self.actor_prime_net.predict(self.er_sp[i].reshape(1,-1))
            
            if (self.er_done[i] == True):
                critic_target[j] = self.er_r[i]
            else:
                critic_target[j] = self.er_r[i] + \
                                   GAMMA*(self.critic_prime_net.predict(self.er_sp[i].reshape(1,-1), action_sp))
        
        self.critic_net.train(Xtrain_state, Xtrain_action, critic_target.reshape(-1,1))
        
        actions_for_derv = self.actor_net.predict(Xtrain_state)
        del_Q_a = self.critic_net.get_derivative_wrt_a(Xtrain_state,
                                                       actions_for_derv)

        del_Q_a = np.array(del_Q_a)
        del_Q_a = del_Q_a.reshape(mb_size, self.action_dim)

        self.actor_net.train(Xtrain_state, del_Q_a)
        #upadte critic prime and actor prime network
        self.update_prime_nets()

    def update_prime_nets(self, tau=TAU):
        actor_weights = self.actor_net.get_weights()
        self.actor_prime_net.set_weights(actor_weights, tau)
        
        critic_weights = self.critic_net.get_weights()
        self.critic_prime_net.set_weights(critic_weights, tau)

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self






