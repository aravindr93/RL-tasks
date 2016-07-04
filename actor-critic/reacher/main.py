"""
    Implementation of the DDPG algorithm of Lillicrap et al. (arXiv:1509.02971)
    22nd June 2016,
    Contributors: Aravind Rajeswaran, Sarvjeet Ghotra, Aravind Srinivas
"""

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
from ActorCritic import *


env = gym.make('Reacher-v1')
state_dim = 11
action_dim = 2
max_steps = min(env.spec.timestep_limit, 250)
print "Time-step limit set to : ", max_steps, " steps."

agent = ActorCritic(state_dim, action_dim, er_size=50000)

state = env.reset()
done_flag = 0
for i in range(5000):

    if (done_flag == True):
        state = env.reset()

    action = env.action_space.sample()
    next_state, reward, done_flag, info = env.step(action)
    agent.append_memory(state, action, reward, next_state, done_flag)
    state = next_state
print "Initial memory built!!"

print "Initial network performance = ", policy_evaluation(agent.learner.actor, env, 50)

start_time = t.time()

wts, best_ep, performance = agent.learn_policy(env,
                                               num_episodes=1500,
                                               learner_update_freq=1,
                                               target_update_freq=5,
                                               epochs=1,
                                               print_freq=50,
                                               save_freq=500,
                                               mb_size=64,
                                               num_evals=20)

end_time = t.time()
print "Total time taken = ", end_time - start_time

plt.plot(performance)
plt.xlabel('Episode')
plt.ylabel('Performance')
plt.title('Learning Curve')
plt.savefig('learning_curve.png')

for _ in range(10):
    inspect_performance(agent.best.actor, env)