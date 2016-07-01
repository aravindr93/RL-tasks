import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

weight_hi = 15
weight_lo = 5
frict_hi = 2.7
frict_lo = 1.8

def draw():
    weight = (weight_hi - weight_lo)*np.random.random_sample() + weight_lo
    friction= (frict_hi - frict_lo)*np.random.random_sample() + frict_lo
    
    x = {'weight': weight,
         'friction': friction}
    
    return x
a = []
for _ in range(0, 100000):
    a.append(draw()['weight'])
    #print(a[-1])

kde = stats.gaussian_kde(a)
# these are the values over wich your kernel will be evaluated
dist_space = np.linspace(weight_lo, weight_hi, 100 )
# plot the results
plt.plot( dist_space, kde(dist_space) )