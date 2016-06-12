import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class SphereEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        mujoco_env.MujocoEnv.__init__(self, 'sphere1.xml', 4)
        utils.EzPickle.__init__(self)
    
    def _step(self, a):
	# Only the control part of cost is implimented here
	# Cost corresponding to deviation from goal position written in agent since simulation needs no access to goal position
	ctrl_cost_coeff = 0.01
        reward_ctrl = -ctrl_cost_coeff*np.square(a).sum()
        reward = reward_ctrl
        self.do_simulation(a, self.frame_skip)
        obs = self._get_obs()
        return obs, reward
    
    def _get_obs(self):
    	pos = self.model.data.qpos.flat[:2]
    	vel = self.model.data.qvel.flat[:2]
    	#print type(pos), pos
    	#print type(vel), vel
    	obs = np.concatenate((pos,vel))
    	return obs
    
    def reset_model(self):
        return self._get_obs()
