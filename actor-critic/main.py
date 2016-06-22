"""
    Implementation of the DDPG algorithm of Lillicrap et al. (arXiv:1509.02971)
    22nd June 2016,
    Contributors: Aravind Rajeswaran, Sarvjeet Ghotra, Aravind Srinivas
"""
#       TODO: Add Noise to the actor for exploration.
#       Add BatchNormalization layers in actor and critic networks.
#       Requires training in GPU since the network is too slow

import gym
import time as t
from agent import ActorCriticAgent
from agent import *
from utils import *
import copy
import pickle


REPLAY_MEMORY = 2000 #According to the paper its 10^6

def main():
    
    env = gym.make('InvertedPendulum-v1')
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    agent = ActorCriticAgent(state_dim, action_dim)
    state = env.reset()
    timestep_limit = min(env.spec.timestep_limit, 20)   # For checking purposes; make it proper for run
    # Initial data build up
    done_flag = 0
    for i in range(REPLAY_MEMORY):

        if (done_flag == True):
            state = env.reset()

        action = env.action_space.sample()
        next_state, reward, done_flag, info = env.step(action)
        agent.append_memory(state, action, reward, next_state, done_flag)
        state = next_state

    print "Initial memory built!!"

    # Initial Training for a few steps
    for _ in range(5):
        agent.update_network()

    print "Initial network performance = ", policy_evaluation(agent, env, 5)
    # =================================================================================

    print "******** Starting learning process *************"
    num_episodes = 10
    update_freq = 1       # update after how many steps (within each episode)
    print_freq = 1        # how often to print (episodes)

    performance = np.zeros(num_episodes)
    best_ep = 0
    best_agent = copy.deepcopy(agent)

    for ep in range(num_episodes):
        done_flag = 0
        state = env.reset()
        time = 0
    
        while (done_flag!=True and time<=timestep_limit):
            action_pred = np.array(agent.actor_net.predict(state.reshape(1,-1)))
            action_pred = action_pred[0]            
            next_state, reward, done_flag, _ = env.step(action_pred)
            agent.append_memory(state, action_pred, reward, next_state, done_flag)
            state = next_state

            #print time, timestep_limit
    
            if (time % update_freq == 0):
                agent.update_network()
    
            time += 1

            performance[ep] = policy_evaluation(agent, env, 2)

        if (ep % print_freq == 0):
            print "Now in episode: ", ep, " of ", num_episodes
            print "Agent performance = ", performance[ep]

        if (performance[ep] > performance[best_ep]):
            best_agent = copy.deepcopy(agent)
            best_ep = ep

    end_time = t.time()
    print "Total time", (end_time - start_time)
    plt.plot(performance[-100:])
    plt.show()
    
    
    # Save agent to file
    with open('objs.pickle', 'wb') as f:
        pickle.dump([best_agent, performance], f)

if __name__ == '__main__':
    main()
