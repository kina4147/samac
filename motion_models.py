import numpy as np
import copy

# x, y, theta, v, theta'
def cv_fx(x, dt):
    px = np.copy(x)
    px[0] = x[0] + dt * x[3] * np.cos(x[2])
    px[1] = x[1] + dt * x[3] * np.sin(x[2])
    px[2] = x[2]
    px[3] = x[3]
    px[4] = 0
    return px

def cv_hx(x):
    return x[0:3]


def ctrv_fx(x, dt):

    return x

def ctrv_hx(x):
    return x[0:3]

