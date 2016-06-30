import keras.backend as K
from keras.layers import Input, Dense
from keras.models import Model, Sequential
import numpy as np

def get_trainable_weights(model):
    """ Get the trainable weights of the model """
    trainable_weights = []
    for layer in model.layers:
        # trainable_weights += keras.engine.training.collect_trainable_weights(layer)
        trainable_weights += layer.trainable_weights
    return trainable_weights

def unpack_theta(model, trainable_weights=None):
    """ Flatten a set of shared variables from model """
    if trainable_weights == None:
        trainable_weights = get_trainable_weights(model)
    x = np.empty(0)
    for param in trainable_weights:
        val = K.eval(param)
        x = np.concatenate([x, val.reshape(-1)])
    return x

def pack_theta(model, theta):
    """ Converts flattened theta back to tensor shape compatible with network """
    weights = []
    idx = 0
    for layer in model.layers:
        layer_weights = []
        for param in layer.get_weights():
            plen = np.prod(param.shape)
            layer_weights.append(np.asarray( theta[idx:(idx+plen)].reshape(param.shape),
                                           dtype=np.float32 ))
            idx += plen
        weights.append(layer_weights)
    weights = [item for sublist in weights for item in sublist]  # change from (list of list) to list
    return weights

def set_model_params(model, theta):
    """ Sets the Keras model params from a flattened numpy array of theta """
    weights = pack_theta(model, theta)
    model.set_weights(weights)
    return model

def compute_gradients(t_loss, param_list, t_inputs, n_inputs):
    """ Computes the derivaties of out (either loss or just Q) wrt the params and returns a flat np array
        NOTE: Make sure that params is a flat list (get using get_trainable_weights for eg)
              Make sure all necessary inputs for the grad computation is present in inputs
              t_inputs is the tensor inputs; and n_inputs is the numeric version """
    # Iterate over each sets of weights in params
    x = np.empty(0)
    for param in param_list:
        c_grad = K.function(t_inputs, K.gradients(t_loss, [param]))
        grad = c_grad(n_inputs) # this is a list
        grad = np.asarray(grad).reshape(-1) # make it into flat array
        x = np.concatenate([x, grad])
    return x