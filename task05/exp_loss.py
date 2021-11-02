"""
author:wjw
des:指数损失loss的计算.
"""
import numpy as np

def exp_loss(y,f):
    return np.exp(-np.dot(y,f)/y.shape[0])


print(exp_loss(np.array([-0.5,1,-0.5]),np.array([-0.1,-0.3,0.4])))
print(np.exp(0.15))