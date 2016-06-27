"""
    Implementation of the DDPG algorithm of Lillicrap et al. (arXiv:1509.02971)
    22nd June 2016,
    Contributors: Aravind Rajeswaran, Sarvjeet Ghotra, Aravind Srinivas
"""

import gym
from ActorCritic import ActorCritic
from ActorCritic import *
from utils import *
from derivative_routines import *
import copy
import pickle
import time as t


REPLAY_MEMORY = 2000  #According to the paper its 10^6

def main():
    
    env = gym.make('Pendulum-v0')
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    agent = ActorCritic(state_dim, action_dim)
    state = env.reset()
    timestep_limit = min(250, env.spec.timestep_limit)
    print "timestep limit set to : ", timestep_limit
    #timestep_limit = env.spec.timestep_limit   # For checking purposes; make it proper for run
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
        agent.update_networks()
        agent.update_target_networks()

    print "Initial network performance = ", policy_evaluation(agent, env, 2)
    # =================================================================================

    print "******** Starting learning process *************"
    num_episodes = 5
    update_freq = 1        # update after how many steps (within each episode)
    print_freq = 1         # how often to print (episodes)

    performance = np.zeros(num_episodes)
    best_ep = 0
    best_agent = copy.deepcopy(agent)

    start_time = t.time()

    for ep in range(num_episodes):
        done_flag = 0
        state = env.reset()
        time = 0
    
        while (done_flag!=True and time<=timestep_limit):
            actor_out = agent.learner.actor.predict(state.reshape(1,-1))[0]
            action = actor_out   # need to add exploration here
            next_state, reward, done_flag, _ = env.step(action)
            agent.append_memory(state, action, reward, next_state, done_flag)
            state = next_state

            if (time % update_freq == 0):
                agent.update_networks(epochs=5)
                #agent.update_target_networks()  --> Ideall I should update here, but it's way too slow.
                #print time, timestep_limit
    
            time += 1

        performance[ep] = policy_evaluation(agent, env, 5)

        # Update the target networks (I'll use a larger tau here)
        agent.update_target_networks(tau=0.01)


        if (ep % print_freq == 0):
            print "Now in episode: ", ep+1, " of ", num_episodes
            print "Agent performance = ", performance[ep]

        if (performance[ep] > performance[best_ep]):
            best_agent = copy.deepcopy(agent)
            best_ep = ep

    end_time = t.time()
    print "Total time", (end_time - start_time)
    plt.plot(performance[-100:])
    plt.show()
    
    
    # Save agent to file (uncomment if you want)
    #with open('objs.pickle', 'wb') as f:
    #    pickle.dump([best_agent, performance], f)

if __name__ == '__main__':
    main()
