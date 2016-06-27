## Actor-Critic RL network for continuous control

Implementation of the DDPG algorithm of Lillicrap et al. (arXiv:1509.02971)

Not yet tested on the Theano backend for Keras. Works correctly when the tensorflow backend is used.

You need to add the following lines of code to this file:
... / lib/python2.7/site-packages/keras/objectives.py

```
def negative_mean_output(y_true, y_pred):
    return -K.mean(y_pred, axis=-1)

# add this line under the aliases
nmo = NMO = negative_mean_output
```
