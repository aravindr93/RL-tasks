"""
    Contains basic utility functions.
"""

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


def policy_evaluation(agent, env, num_episodes=10, max_steps=2000):
    "Evaluate quality of policy by performing rollouts"
    evaluation = 0.0
    for _ in range(num_episodes):
        state = env.reset()
        episode_reward = 0.0
        done = 0
        time = 0
        while (done != True and time < max_steps):
            Q = np.array(agent.actor_prime_net.predict(state.reshape(1, -1)))
            #a = int(np.argmax(Q.ravel()))
            sp, r, done, info = env.step(Q[0])
            env.render()
            if (done != True):
                episode_reward += r
            state = sp
            time += 1
        evaluation = evaluation + episode_reward/num_episodes

    env.reset()
    return evaluation    

def inspect_performance(agent, env):
    for _ in range(20):
        s = env.reset()
        time = 0
        done = 0

    while (done != True and time < 1500):
        env.render()
        Q = np.array(agent.net.predict(s.reshape(1,-1)))
        pi, a = epsilon_greedy(Q.ravel(),eps=0)
        sp, r, done, info = env.step(a)
        s = sp

