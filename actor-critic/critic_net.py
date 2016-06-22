"""
    This file contains the critic network.
"""
import numpy as np
import tensorflow as tf
import math

LEARNING_RATE = 0.001
TAU = 0.001
BATCH_SIZE = 64

class Critic:
    
    def __init__(self, states_dim, action_dim):
        self.g = tf.Graph()
        
        with self.g.as_default():
            self.sess = tf.InteractiveSession()
            
            self.W1, self.B1, self.W2, self.W2_action, self.B2, self.W3, self.B3, \
            self.curr_state, self.action, self.critic_net = self.create_net(states_dim, action_dim)
            
            #cost
            #self.del_Q_action = tf.placeholder("float", [None, action_dim])
            self.critic_param = [self.W1, self.B1, self.W2, self.W2_action, self.B2, self.W3, self.B3]
            self.q_value = tf.placeholder("float", [None, 1])
            #can be converted to a function
            self.l2_regularizer_loss = tf.nn.l2_loss(self.W1) + tf.nn.l2_loss(self.W2)\
                + tf.nn.l2_loss(self.W2_action) +  tf.nn.l2_loss(self.W3)+tf.nn.l2_loss(self.B1)+\
                tf.nn.l2_loss(self.B2)+tf.nn.l2_loss(self.B3) 
            self.cost = tf.pow((self.critic_net-self.q_value), 2)/BATCH_SIZE +1e-2*self.l2_regularizer_loss
            self.optimizer = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE).minimize(self.cost)
            
            self.act_grad_v = tf.gradients(self.critic_net, self.action)
            #this is just divinding by batch size
            self.wrt_action_gradients = [self.act_grad_v[0]/tf.to_float(tf.shape(self.act_grad_v[0])[0])]
            
            #initialize all the variables
            self.sess.run(tf.initialize_all_variables())
    
    def create_net(self, num_states, num_actions):
        """Critic Network that maps states and actions to Q value"""
        
        N_HIDDEN_1 = 400
        N_HIDDEN_2 = 400
        curr_state = tf.placeholder("float",[None,num_states])
        action = tf.placeholder("float",[None,num_actions])    
    
        W1 = tf.Variable(tf.random_uniform([num_states,N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)))
        B1 = tf.Variable(tf.random_uniform([N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)))
        W2 = tf.Variable(tf.random_uniform([N_HIDDEN_2,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+num_actions),1/math.sqrt(N_HIDDEN_1+num_actions)))    
        W2_action = tf.Variable(tf.random_uniform([num_actions,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+num_actions),1/math.sqrt(N_HIDDEN_1+num_actions)))    
        B2 = tf.Variable(tf.random_uniform([N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1+num_actions),1/math.sqrt(N_HIDDEN_1+num_actions))) 
        W3 = tf.Variable(tf.random_uniform([N_HIDDEN_2,1],-0.003,0.003))
        B3= tf.Variable(tf.random_uniform([1],-0.003,0.003))
    
        H1 =tf.nn.softplus(tf.matmul(curr_state, W1)+ B1)
        H2 =tf.nn.tanh(tf.matmul(H1, W2)+tf.matmul(action, W2_action)+B2)
            
        critic_net = tf.matmul(H2,W3)+B3
        
        return W1, B1, W2, W2_action, B2, W3, B3, curr_state, action, critic_net

    def predict(self, state, action):
        
        return self.sess.run(self.critic_net, feed_dict={self.curr_state: state,\
                self.action: action})

    def train(self, state, action, y):
        
        self.sess.run(self.optimizer, feed_dict={self.curr_state: state,
                                    self.action: action, self.q_value: y})
    
    
    def get_weights(self):
        
        return [self.sess.run(x) for x in self.critic_param]
    
    
    def set_weights(self, weights, tau):
        for i in range(len(weights)):
            self.sess.run(self.critic_param[i].assign(tau*weights[i]+(1-tau)*self.critic_param[i]))
    
    #testing required
    def get_derivative_wrt_a(self, state, action):
        
        return self.sess.run(self.wrt_action_gradients, feed_dict={self.curr_state:state, \
                                            self.action: action})


















            
