## Draw(), set_param(param), step(action, param)

These are the changed files to enable parameter variation drawing and passing paramter argument to step() in openai gym.

## Paths to the files
core.py - gym/gym/

pendulum.py - gym/gym/envs/classic_control/

bipedal_walker.py - gym/gym/envs/box2d
Variable paramters in BipedalWalker-v2:
    Static: motor_torque and friction
    dynamic: speed_hip, speed_knee
    potential variable that are likely to be variable have not been included, yet: lindar_range, desities, motor speed, 
    shape of the walker like leg's length, head etc.
        
Draw_testing is just for testing the distibution curve of the
draw(). 

For time being it draws from continous distribution.
Later will change as per the most appropriate one for the 
parameter variations.
