"""
    This file contains the actor network.
"""

import tensorflow as tf
import math

LEARNING_RATE = 0.0001
TAU = 0.001

class Actor:
    
    def __init__(self, states_dim, action_dim):
        self.g = tf.Graph()
        
        with self.g.as_default():
            self.sess = tf.InteractiveSession()
            
            self.W1, self.B1, self.W2, self.B2, self.W3, self.B3, \
            self.curr_state, self.actor_net = self.create_net(states_dim, action_dim)
            
            self.del_Q_action = tf.placeholder("float", [None, action_dim]) #Derivative of Critic wrt to actions
            self.actor_param = [self.W1, self.B1, self.W2, self.B2, self.W3, self.B3]
            
            #why not dividing with batch_size: first because I am calculating it one by one 
            #and doing mean in the Actor code
            self.param_gradient = tf.gradients(self.actor_net, self.actor_param, \
                grad_ys=-self.del_Q_action) #grad_ys gives weights to the derivatives
            self.optimizer = tf.train.AdamOptimizer(LEARNING_RATE).apply_gradients(\
                    zip(self.param_gradient,self.actor_param))
            #initialize all the variables
            self.sess.run(tf.initialize_all_variables())
    
    def create_net(self, num_states, num_actions):
        """Actor Network that maps states to actions"""
        N_HIDDEN_1 = 400
        N_HIDDEN_2 = 300
        
        #See the supplementary section of the page it describes the architecture
        curr_state = tf.placeholder("float", [None, num_states])
        W1=tf.Variable(tf.random_uniform([num_states,N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)))
        B1=tf.Variable(tf.random_uniform([N_HIDDEN_1],-1/math.sqrt(num_states),1/math.sqrt(num_states)))
        W2=tf.Variable(tf.random_uniform([N_HIDDEN_1,N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1),1/math.sqrt(N_HIDDEN_1)))
        B2=tf.Variable(tf.random_uniform([N_HIDDEN_2],-1/math.sqrt(N_HIDDEN_1),1/math.sqrt(N_HIDDEN_1)))
        W3=tf.Variable(tf.random_uniform([N_HIDDEN_2,num_actions],-0.003,0.003))
        B3=tf.Variable(tf.random_uniform([num_actions],-0.003,0.003))
        
        H1 = tf.nn.softplus(tf.matmul(curr_state, W1)+B1)
        H2 = tf.nn.tanh(tf.matmul(H1,W2)+B2)
        actor_net = tf.matmul(H2,W3) + B3
        actor_net = tf.nn.tanh(actor_net)
        #p = tf.Variable(5.0)
        #q = tf.Variable(10.0)
        return W1, B1, W2, B2, W3, B3, curr_state, actor_net
    
    def predict(self, state):
        
        return self.sess.run(self.actor_net, feed_dict={self.curr_state: state})

    def train(self, state, del_Q_a):
        
        self.sess.run(self.optimizer, feed_dict={self.curr_state: state,
                                                 self.del_Q_action: del_Q_a})
    
    def get_weights(self):
        
        return [self.sess.run(x) for x in self.actor_param]
    
    def set_weights(self, weights, tau):
        
        for i in range(len(weights)):
            self.sess.run(self.actor_param[i].assign(tau*weights[i]+(1-tau)*self.actor_param[i]))
    


















            