"""
    Contains basic utility functions.
"""

import keras.backend as K
from keras.layers import Input, Dense
from keras.models import Model, Sequential
import numpy as np


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


def policy_evaluation(actor, env, num_episodes=10, max_steps=250):
    "Evaluate quality of policy by performing rollouts | actor should predict action given state"
    evaluation = 0.0
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        done = 0
        time = 0
        while (done != True and time < max_steps):
            action = actor.predict(state.reshape(1,-1))[0]
            sp, r, done, info = env.step(action)
            if (done != True):
                episode_reward += r
            state = sp
            time += 1
        evaluation = evaluation + episode_reward/num_episodes

    env.reset()
    return evaluation    

def inspect_performance(actor, env, num_episodes=1):
    max_steps = env.spec.timestep_limit
    for _ in range(num_episodes):
        s = env.reset()
        time = 0
        done = 0
        while (done != True and time < max_steps):
            env.render()
            a = np.array(actor.predict(s.reshape(1, -1)))
            sp, r, done, info = env.step(a)
            s = sp
            time += 1
        print "Completed in: ", time, " (max allowed = ", max_steps, ")"

def get_trainable_weights(model):
    """ Get the trainable weights of the model """
    trainable_weights = []
    for layer in model.layers:
        # trainable_weights += keras.engine.training.collect_trainable_weights(layer)
        trainable_weights += layer.trainable_weights
    return trainable_weights

def unpack_theta(model, trainable_weights=None):
    """ Flatten a set of shared variables from model """
    if trainable_weights == None:
        trainable_weights = get_trainable_weights(model)
    x = np.empty(0)
    for param in trainable_weights:
        val = K.eval(param)
        x = np.concatenate([x, val.reshape(-1)])
    return x

def pack_theta(model, theta):
    """ Converts flattened theta back to tensor shape compatible with network """
    weights = []
    idx = 0
    for layer in model.layers:
        layer_weights = []
        for param in layer.get_weights():
            plen = np.prod(param.shape)
            layer_weights.append(np.asarray( theta[idx:(idx+plen)].reshape(param.shape),
                                           dtype=np.float32 ))
            idx += plen
        weights.append(layer_weights)
    weights = [item for sublist in weights for item in sublist]  # change from (list of list) to list
    return weights

def set_model_params(model, theta):
    """ Sets the Keras model params from a flattened numpy array of theta """
    weights = pack_theta(model, theta)
    model.set_weights(weights)
    return model

def get_exploration(env, ep, num_episodes, high=0.5, low=0.1):
    a_high = np.asarray(env.action_space.high)
    a_low = np.asarray(env.action_space.low)
    a_dim = len(a_high)
    slp = (high-low)/num_episodes
    val = (high - slp*ep)*(a_high-a_low)
    return a_low + val*np.random.rand(a_dim)