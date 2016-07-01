# Draw(), set_param(param), step(action, param)

These are the changed files to enable parameter variation drawing and passing paramter argument to step() in openai gym.

# Paths to the files
core.py - gym/gym/
pendulum.py - gym/gym/envs/classic_control/
Dra_testing is just for testing the distibution curve of the
draw(). For time being it draws from continous distribution.
Later will change as per the most appropriate one for the 
parameter variations.
