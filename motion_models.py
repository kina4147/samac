import numpy as np
# CV w/ heading : x, y, rz, v, vrz
def cv_fx(x, dt):
    px = np.copy(x)
    px[0] = x[0] + dt * x[3] * np.cos(x[2])
    px[1] = x[1] + dt * x[3] * np.sin(x[2])
    px[2] = x[2]
    px[3] = x[3]
    px[4] = 0
    return px


# CTPV w/ heading : x, y, rz, v, vrz
def ctpv_fx(x, dt):
    px = np.copy(x)
    if np.absolute(x[4]) > 0.0001:
        px[0] = x[0] + 2.0 * x[3] / x[4] * np.sin(0.5 * x[4] * dt) * np.cos(x[2] + 0.5 * x[4] * dt)
        px[1] = x[1] + 2.0 * x[3] / x[4] * np.sin(0.5 * x[4] * dt) * np.sin(x[2] + 0.5 * x[4] * dt)
        px[2] = x[2] + dt * px[4]
        px[3] = x[3]
        px[4] = x[4]
    else:
        px[0] = x[0] + dt * x[2] * np.cos(x[3])
        px[1] = x[1] + dt * x[2] * np.sin(x[3])
        px[2] = x[2] + dt * px[4]
        px[3] = x[3]
        px[4] = x[4]
    return px


# CTRV w/ heading : x, y, rz, v, vrz
def ctrv_fx(x, dt):
    px = np.copy(x)
    if np.absolute(x[4]) > 0.0001:
        px[0] = x[0] + (x[3] / x[4]) * (np.sin(x[2] + dt * x[4]) - np.sin(x[2]))
        px[1] = x[1] + (x[3] / x[4]) * (-np.cos(x[2] + dt * x[4]) + np.cos(x[2]))
        px[2] = x[2] + dt * px[4]
        px[3] = x[3]
        px[4] = x[4]
    else:
        px[0] = x[0] + dt * x[2] * np.cos(x[3])
        px[1] = x[1] + dt * x[2] * np.sin(x[3])
        px[2] = x[2] + dt * px[4]
        px[3] = x[3]
        px[4] = x[4]
    return px


def rm_fx(x, dt):
    px = np.copy(x)
    px[0] = x[0]
    px[1] = x[1]
    px[2] = x[2]
    px[3] = 0
    px[4] = 0
    return px

# old => 0: x, 1: y, 2: x', 3: y', 4: v, 5: theta, 6: theta'
# new => 0: x, 1: y, 2: theta, 3: v, 4: theta', 5: x', 6: y'
def aug_ctcv_fx(x, dt):
    px = x.copy()
    if np.absolute(x[4]) > 0.0001:
        px[0] = x[0] + x[5]/x[4]*np.sin(x[4]*dt) - x[6]/x[4]*(1 - np.cos(x[4]*dt))
        px[1] = x[1] + x[5]/x[4]*(1 - np.cos(x[4]*dt)) + x[6]/x[4]*np.sin(x[4]*dt)
        px[4] = x[4]
        px[5] = x[5] * np.cos(x[4]*dt) - x[6] * np.sin(x[4]*dt)
        px[6] = x[5] * np.sin(x[4]*dt) + x[6] * np.cos(x[4]*dt)
    else:
        px[0] = x[0] + dt * x[5]
        px[1] = x[1] + dt * x[6]
        px[4] = x[4]
        px[5] = x[5] * np.cos(x[4]*dt) - x[6] * np.sin(x[4]*dt)
        px[6] = x[5] * np.sin(x[4]*dt) + x[6] * np.cos(x[4]*dt)

    px[2] = np.sqrt(px[5]**2 + px[6]**2)
    px[3] = np.arctan2(px[6], px[5])
    return px

def aug_ctpv_fx(x, dt):
    px = x.copy()
    if np.absolute(x[4]) > 0.0001:
        px[0] = x[0] + 2.0 * x[3] / x[4] * np.sin(0.5 * x[4] * dt) * np.cos(x[2] + 0.5 * x[4] * dt)
        px[1] = x[1] + 2.0 * x[3] / x[4] * np.sin(0.5 * x[4] * dt) * np.sin(x[2] + 0.5 * x[4] * dt)
        px[2] = x[2] + dt * x[4]
        px[3] = x[3]
        px[4] = x[4]
    else:
        px[0] = x[0] + dt * x[3] * np.cos(x[2])
        px[1] = x[1] + dt * x[3] * np.sin(x[2])
        px[2] = x[2] + dt * x[4]
        px[3] = x[3]
        px[4] = x[4]

    px[5] = px[3]*np.cos(px[2])
    px[6] = px[3]*np.sin(px[2])
    return px


# def aug_ctrv_fx(x, dt):
#     px = x.copy()
#     if np.absolute(x[6]) > 0.01:
#         px[0] = px[0] + (px[4] / px[6]) * (np.sin(px[5] + dt * px[6]) - sin(px[5]))
#         px[1] = px[1] + (px[4] / px[6]) * (-np.cos(px[5] + dt * px[6]) + cos(px[5]))
#         px[4] = px[4]
#         px[5] = px[5] + dt * px[6]
#         px[6] = px[6]
#     else:
#         px[0] = px[0] + dt * px[4] * np.cos(px[5])
#         px[1] = px[1] + dt * px[4] * np.sin(px[5])
#         px[4] = px[4]
#         px[5] = px[5]
#         px[6] = px[6]
#     #     x[2] = x[4]*cos(x[5])
#     #     x[3] = x[4]*sin(x[5])
#     return px

def hx(x):
    return x[0:3]


def residual(x, y):
    diff = x - y
    while diff[2] > np.pi:
        diff[2] -= 2 * np.pi
    while diff[2] < -np.pi:
        diff[2] += 2 * np.pi
    return diff
