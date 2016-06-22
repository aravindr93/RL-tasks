"""
    Reinforcement learning agent based on Deep Deterministic 
    Poilcy Gradient (DDPG). For continous action spaces.
    Sarvjeet Singh Ghotra, 21th June 2016
"""

#TODO: Add Noise to the actor for exploration.
#      Add BatchNormalization layers in actor and critic networks.

import gym
import time as t
from agent import ActorCriticAgent
from agent import *
from utils import *


REPLAY_MEMORY = 20000 #According to the paper its 10^6

def main():
    
    env = gym.make('BipedalWalker-v2')
    action_dim = env.action_space.shape[0]
    state_dim = env.observation_space.shape[0]
    agent = ActorCriticAgent(state_dim, action_dim)
    state = env.reset()
    timestep_limit = env.spec.timestep_limit
    # Initial data build up
    done_flag = 0
    for i in range(REPLAY_MEMORY):
        # Reset env once in a while
        if (done_flag == True):
            state = env.reset()
    
        action = env.action_space.sample()
        next_state, reward, done_flag, _ = env.step(action)
        # if(done_flag):
        # reward=-10
    
        agent.append_memory(state, action, reward, next_state, done_flag)
        state = next_state
    
    print "Initial memory built!!"
    
    
    # Initial Training
    for _ in range(2):
        agent.update_network()
    
    #agent.update_network()
    #print "Initial network loss = ", history.history
    # =================================================================================
    
    print "******** Starting learning process *************"
    num_episodes = 500
    update_freq = 1      # update after how many steps (within each episode)
    print_freq = 40      # how often to print (episodes)
    
    performance = np.zeros(num_episodes)
    error_decay = np.zeros(num_episodes)
    
    for ep in range(num_episodes):
        done_flag = 0
        state = env.reset()
        time = 0
    
        while (done_flag!=True and time<=timestep_limit):
            action_pred = np.array(agent.actor_net.predict(state.reshape(1,-1)))
            # pi, action = softmax_policy(Q_pred.ravel())
            #pi, action = softmax_policy(action_pred.ravel())
            action_pred = action_pred[0]            
            next_state, reward, done_flag, _ = env.step(action_pred)
            print action_pred
            agent.append_memory(state, action_pred, reward, next_state, done_flag)
            state = next_state
    
            if (time % update_freq == 0):
                agent.update_network()
    
            time += 1
        
        performance[ep] = policy_evaluation(agent, env, 5)
        #error_decay[ep] = agent.net.loss_
        
        if (ep % print_freq == 0):
            print "Now in episode: ", ep, " of ", num_episodes
            print "Agent performance = ", performance[ep]
            print "Size of er: ", agent.len_of_replay()
    
    end_time = t.time()
    print "Total time", (end_time - start_time)
    plt.plot(performance[-100:])
    plt.show()
    
    
    # Save agent to file
    with open('objs.pickle', 'w') as f:
        pickle.dump(agent, f)

if __name__ == '__main__':
    main()