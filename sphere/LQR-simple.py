''' 
	General LQR controller with local model learning
	Currently implimentable for go-to-goal tasks
	For tracking trajector need to write function to compute tracking error

	Aravind Rajeswaran, 12th June 2016
	IIT Madras, Chennai, India
'''

'''
	Current Bug: In the video, MuJoCo also renders the small perturbations
	used to calculate the local models. Need to figure out how to store only
	those frames required for the movie.
	This is purely cosmetic, the code and algorithm calculates the correct
	control decisions as can be checked from plots (which contain only required frames)
'''

import gym
import time
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
#import control as ctrl  	# optional, can be downloaded from Richard Murray's website (Caltech CDS)
import scipy.linalg

env = gym.make('Sphere-v0')

def dlqr(A,B,Q,R):
	# solves the discrete time LQR problem 
	# sub-routine for solving Algebraic Ricatti equation available in scipy.linalg
    X = np.matrix(scipy.linalg.solve_discrete_are(A, B, Q, R))
    #compute the LQR gain
    K = np.matrix(scipy.linalg.inv(B.T*X*B+R)*(B.T*X*A))
    eigVals, eigVecs = scipy.linalg.eig(A-B*K)
    
    K = np.asarray(K)
    return K, X, eigVals

class LQR(object):
    """LQR class for implimenting control"""
    def __init__(self, gamma):
        # Initialize LQR with random model parameters
        self.nx = env.model.nq+env.model.nv   # number of states is pos + velocity
        self.nu = 2		# number of actuators
        self.A = np.random.rand(self.nx, self.nx)
        self.B = np.random.rand(self.nx, self.nu)  # 2 actuators
        self.Q = np.diag([1, gamma, gamma**2, gamma**3])
        self.R = np.eye(self.nu)
        self.K = np.random.rand(self.nu, self.nx)
        
        # book-keeping of time and states
        self.time_step = []
        self.state_memory = []

    def move_to(self, state):
    	# simple function to move to a given state
    	pos = state[0:2]
    	vel = state[2:4]
    	env.set_state(pos,vel)
        
    def plot_data(self, state):
        if not self.time_step:
        	time=0
        else:
        	time=self.time_step[-1]+1
        self.time_step.append(time)
        self.state_memory.append(state)
        
    def controller(self, state):
        # important function
        # calculates the control action given the state
    
        # step 1: learn local model A and B
        # step 2: using A, B, Q, R find the control gain K
        # step 3: use state feedback form to calculate control = -K*state

        # learn model only once in a while
        if(self.time_step[-1]%50 == 0):
        	self.A,self.B = self.local_learn(state)  # locally learn model around state
        	self.K, Ig1, Ig2 = dlqr(self.A, self.B, self.Q, self.R)
        return self.K

    def local_learn(self, state):
    	A = self.A
    	B = self.B

    	x0 = state.copy()
    	u0 = np.array([0,0])

    	epsx = 0.1
    	epsu = 1             # Order of magnitude is a little tricky!

    	for i in range(self.nx):
    		x = x0.copy()
    		
    		x[i] += epsx; 	self.move_to(x)
    		x_inc = env.step(u0)[0]
    		x = x0.copy()
    		
    		x[i] -= epsx; 	self.move_to(x)
    		x_dec = env.step(u0)[0]

    		A[:,i] = (x_inc - x_dec)/(2*epsx)
    	# ----------
    	
    	for j in range(self.nu):
    		u = u0.copy()

    		u[j] += epsu;	self.move_to(x0)
    		x_inc = env.step(u)[0]
    		u = u0.copy()

    		u[j] -= epsu;	self.move_to(x0)
    		x_dec = env.step(u)[0]

    		B[:, j] = (x_inc - x_dec)/(2*epsu)

    		#print x_inc
    		#print x_dec

    	self.move_to(x0)

    	return A,B


agent = LQR(0.9)
max_u = 10

outdir = '/tmp/LQR'
env.monitor.start(outdir, force=True)
# first step is to reset environment (mujoco requirement, so do it!)
env.reset()
# select initial position and velocity
qinit = np.array([3, -9])
vinit = np.array([0, 0])
# select goal state
goal_state = np.array([-7, 3, 0, 0])

# set environment to initial state
env.set_state(qinit,vinit)
state = env._get_obs()

for _ in range(100):
	env.render()

	agent.plot_data(state)
	# calculate the controller
	K = agent.controller(state)
	# calculate error signal
	err = state-goal_state
	# compute controls
	u = -np.dot(K,err)

	u = np.clip(u, -max_u, max_u)

	state = env.step(u)[0]

env.monitor.close()    

# Plots
time = np.array(agent.time_step)
x = np.array( [item[0] for item in agent.state_memory] )
y = np.array( [item[1] for item in agent.state_memory] )
plt.plot(time, x)
plt.plot(time, y)
plt.show()
plt.plot(x,y)
plt.show()
