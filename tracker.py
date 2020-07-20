import numpy as np
from filterpy.kalman import unscented_transform
from filterpy.kalman import (KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter)
from filterpy.kalman import (unscented_transform, MerweScaledSigmaPoints, JulierSigmaPoints)
from filterpy.kalman import IMMEstimator
import motion_models as mm
from covariance import Covariance
from utils import angle_in_range

from collections import namedtuple
SAMACTracker = namedtuple('SAMACTracker', ['info', 'mtracker', 'atracker'])
class TrackerGenerator(object):
    def __init__(self, tracking_name='car'):
        self.tracking_name = tracking_name
        # x, y, z, yaw, w, l, h, x', y', z', yaw', w', l', h'
        self.association_metrics = ['mdist'] # 'iou' or 'mdist' 'mahalanobis distance'
        self.mstate_on = np.array([True, True, True])
        self.dim_m_x = 5
        self.dim_m_z = 3
        self.astate_on = np.array([True, False, False, True])
        self.dim_a_x = 4
        self.dim_a_z = 4
        self.imm_on = False
        self.yaw_pos = 2
        # for tracking_name in NUSCENES_TRACKING_NAMES:
        if tracking_name == 'bicycle': # holonomic
            self.head_appearance_on = False
            self.head_motion_on = True
            self.association_metrics = ['mdist']
        elif tracking_name == 'bus': # motional
            self.head_appearance_on = True
            self.head_motion_on = True
            self.association_metrics = ['mdist']
        elif tracking_name == 'car': # motional
            self.head_appearance_on = True
            self.head_motion_on = True
            self.association_metrics = ['mdist']
        elif tracking_name == 'motorcycle':
            self.head_appearance_on = False
            self.head_motion_on = False
            self.association_metrics = ['mdist']
        elif tracking_name == 'pedestrian':
            self.head_appearance_on = False
            self.head_motion_on = True
            self.association_metrics = ['mdist']
        elif tracking_name == 'trailer':
            self.head_appearance_on = False
            self.head_motion_on = True
            self.association_metrics = ['mdist']
        elif tracking_name == 'truck':
            self.head_appearance_on = True
            self.head_motion_on = True
            self.association_metrics = ['mdist']
        else:
            assert False

        if tracking_name == 'bicycle' or tracking_name == 'motorcycle':
            self.astate_on = np.array([True, True, True, True])
        elif tracking_name == 'pedestrian':
            self.astate_on = np.array([True, False, False, True])
        elif tracking_name == 'bus' or tracking_name == 'trailer' or tracking_name == 'truck':
            self.astate_on = np.array([True, True, False, True])
        elif tracking_name == 'car':
            self.astate_on = np.array([True, True, True, True])
        else:
            assert False

        # reference
        # self.head_appearance_on = True
        # self.head_motion_on = False
        # self.association_metrics = ['mdist']
        # self.astate_on = np.array([True, True, True, True])

        cov = Covariance(1)
        self.mmodel = {}
        if self.head_appearance_on:  # x, y, theta
            self.mstate_on = np.array([True, True, True])
            self.dim_m_z = np.count_nonzero(self.mstate_on)
            if self.head_motion_on:  # x, y, theta, v, theta'
                self.linear_motion = False
                self.dim_m_x = 5 # x, y, theta, v, theta'
                self.dim_m_z = 3 # x, y, theta, v, theta'
                self.mmodel['fx'] = mm.cv_fx # cv vs. ctrv
                self.mmodel['hx'] = mm.hx_3
                self.mmodel['residual'] = mm.residual
                self.mmodel['P'] = cov.m_head_P[tracking_name][:self.dim_m_x, :self.dim_m_x]
                self.mmodel['Q'] = cov.m_head_Q[tracking_name][:self.dim_m_x, :self.dim_m_x]
                self.mmodel['R'] = cov.m_R[tracking_name][:self.dim_m_z, :self.dim_m_z]
            else:
                self.linear_motion = True
                self.dim_m_x = 6 # x, y, theta, x', y', theta'
                self.dim_m_z = 3 # x, y, theta, x', y', theta'
                self.mmodel['P'] = cov.m_no_head_P[tracking_name][:self.dim_m_x, :self.dim_m_x]
                self.mmodel['Q'] = cov.m_no_head_Q[tracking_name][:self.dim_m_x, :self.dim_m_x]
                self.mmodel['R'] = cov.m_R[tracking_name][:self.dim_m_z, :self.dim_m_z]
        else:
            self.mstate_on = np.array([True, True, False])
            self.yaw_pos = -1
            self.num_mstate = np.count_nonzero(self.mstate_on)
            if self.head_motion_on:  # x, y, theta, v, theta'
                self.linear_motion = False
                self.dim_m_x = 5 # x, y, theta, v, theta'
                self.dim_m_z = 2 # x, y, theta, v, theta'
                self.mmodel['fx'] = mm.cv_fx # cv vs. ctrv
                self.mmodel['hx'] = mm.hx_2
                self.mmodel['residual'] = mm.residual
                self.mmodel['P'] = cov.m_head_P[tracking_name][:self.dim_m_x, :self.dim_m_x]
                self.mmodel['Q'] = cov.m_head_Q[tracking_name][:self.dim_m_x, :self.dim_m_x]
                self.mmodel['R'] = cov.m_R[tracking_name][:self.dim_m_z, :self.dim_m_z]
            else:
                self.linear_motion = True
                self.dim_m_x = 5 # x, y, theta, x', y'
                self.dim_m_z = 3 # x, y, theta, x', y'
                self.mmodel['P'] = cov.m_no_head_P[tracking_name][:self.dim_m_x, :self.dim_m_x]
                self.mmodel['Q'] = cov.m_no_head_Q[tracking_name][:self.dim_m_x, :self.dim_m_x]
                self.mmodel['R'] = cov.m_R[tracking_name][:self.dim_m_z, :self.dim_m_z]

        # z, w, l, h
        self.dim_a_x = 4
        self.dim_a_z = 4 # np.count_nonzero(self.astate_on)
        self.amodel = {}
        self.amodel['P'] = cov.a_P[tracking_name][0:self.dim_a_x, 0:self.dim_a_x]
        self.amodel['Q'] = cov.a_Q[tracking_name][0:self.dim_a_x, 0:self.dim_a_x]
        self.amodel['R'] = cov.a_R[tracking_name][0:self.dim_a_z, 0:self.dim_a_z]

        self.state_on = np.concatenate((self.mstate_on, self.astate_on))
        self.state_cov_on = self.state_on.reshape(-1, 1).dot(self.state_on.reshape(-1, 1).T)
        self.dim_z = np.count_nonzero(self.state_on)
        self.mdist_thres = cov.mdist_threshold[self.dim_z]

    def generate_tracker(self, z, info, track_score=0.0):
        # z: x, y, yaw, z, l, w, h
        m_z = np.copy(z[0:3])
        a_z = np.copy(z[3:7])
        # Motion Tracker Initialization
        if self.linear_motion: # x, y, theta, x', y', theta' or x, y, theta, x', y'
            mtracker = KFTracker(z=m_z, dim_x=self.dim_m_x, dim_z=self.dim_m_z, dim_out=3, yaw_pos=2, model=self.mmodel, tracking_name=self.tracking_name)
        else:
            if not self.imm_on:
                mtracker = UKFTracker(z=m_z, dim_x=self.dim_m_x, dim_z=self.dim_m_z, dim_out=3, yaw_pos=2, model=self.mmodel, tracking_name=self.tracking_name)
            else:
                models = []
                # # cv_fx
                model_1 = {'fx': mm.cv_fx, 'hx': mm.hx_3, 'residual': mm.residual, 'mode_prob': 0.3, 'transition_prob': 0.96, 'P': self.mmodel['P'], 'Q': self.mmodel['Q'], 'R':self.mmodel['R']}
                models.append(model_1)
                # # ctrv_fx
                model_1 = {'fx': mm.ctrv_fx, 'hx': mm.hx_3, 'residual': mm.residual, 'mode_prob': 0.3, 'transition_prob': 0.96, 'P': self.mmodel['P'], 'Q': self.mmodel['Q'], 'R':self.mmodel['R']}
                models.append(model_1)
                # # rm_fx
                model_1 = {'fx': mm.rm_fx, 'hx': mm.hx_3, 'residual': mm.residual, 'mode_prob': 0.3, 'transition_prob': 0.96, 'P': self.mmodel['P'], 'Q': self.mmodel['Q'], 'R':self.mmodel['R']}
                models.append(model_1)
                mtracker = IMMTracker(z=m_z, dim_x=self.dim_m_x, dim_z=self.dim_m_z, dim_out=3, yaw_pos=2, model=models, tracking_name=self.tracking_name)


        # Appearance Tracker Initialization
        atracker = KFTracker(z=a_z, dim_x=self.dim_a_x, dim_z=self.dim_a_z, dim_out=self.dim_a_z, yaw_pos=-1, model=self.amodel, tracking_name=self.tracking_name)

        # Track Info Initialization
        info = TrackerInfo(state_on=np.concatenate((self.mstate_on, self.astate_on)), info=info, track_score=track_score, tracking_name=self.tracking_name)

        # tracker = {'info': info, 'mtracker': mtracker, 'atracker': atracker}
        tracker = SAMACTracker(info, mtracker, atracker)
        return tracker



class TrackerInfo(object):
    count = 0
    def __init__(self, info=None, state_on=np.ones(7, dtype=bool), track_score=None, tracking_name='car'):
        self.id = TrackerInfo.count
        TrackerInfo.count += 1
        self.time_since_update = 0
        self.hits = 1  # number of total hits including the first detection
        self.hit_streak = 0  # number of continuing hit considering the first detection
        self.first_continuing_hit = 0
        self.still_first = True
        self.age = 0
        self.info = info  # other info
        self.track_score = track_score
        self.tracking_name = tracking_name
        self.state_on = state_on
        self.tracking = False
        self.erase = False

    def predict(self):
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
            self.still_first = False
        self.time_since_update += 1

    def update(self, info, track_score):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1  # number of continuing hit
        if self.still_first:
            self.first_continuing_hit += 1  # number of continuing hit in the fist time

        self.info = info
        self.track_score = track_score


class Tracker(object):
    def __init__(self, dim_x, dim_z, dim_out, yaw_pos, model=None):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dim_out = dim_out
        self.yaw_pos = yaw_pos
        self.model = model
        self.yaw_pos = yaw_pos
        self.history = []
        self.filter = None

    def predict(self, dt=1.0):
        self.filter.predict(dt)

    def update(self, z, info):
        self.filter.update(z=z)

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return self.filter.x[:self.dim_out].reshape((-1, 1))

class KFTracker(Tracker): # x, y, x', y'
    def __init__(self, z, dim_x, dim_z, dim_out, yaw_pos=2, model=None, tracking_name='car'):
        super().__init__(dim_x, dim_z, dim_out, yaw_pos, model)
        self.filter = KalmanFilter(dim_x=self.dim_x, dim_z=self.dim_z)
        self.filter.P = model['P']
        self.filter.Q = model['Q']
        self.filter.R = model['R']
        self.filter.F = np.eye(self.dim_x) + np.eye(self.dim_x, k=self.dim_z)
        self.filter.H = np.eye(self.dim_z, self.dim_x)
        if self.yaw_pos > -1: # head inclusion
            z[self.yaw_pos] = angle_in_range(z[self.yaw_pos])
        self.filter.x[:self.dim_out, :] = z.reshape((-1, 1))

    def predict(self, dt=1.0):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.filter.predict(dt)
        if self.yaw_pos > -1: # head inclusion
            self.filter.x[self.yaw_pos] = angle_in_range(self.filter.x[self.yaw_pos])
        self.history.append(self.filter.x)

        PHT = np.dot(self.filter.P, self.filter.H.T)
        S = np.dot(self.filter.H, PHT) + self.filter.R
        pred_z = np.dot(self.filter.H, self.filter.x)
        if self.yaw_pos > -1:
            pred_z[self.yaw_pos] = angle_in_range(pred_z[self.yaw_pos])

        if self.dim_z < self.dim_out:
            out_pred_z = np.zeros((self.dim_out, 1))
            out_S = np.zeros((self.dim_out, self.dim_out))
            out_pred_z[:self.dim_z] = pred_z
            out_S[:self.dim_z, :self.dim_z] = S
            return out_pred_z, out_S
        return pred_z, S

    def update(self, z):
        self.history = []
        if self.yaw_pos > -1: # head inclusion
            z[self.yaw_pos] = angle_in_range(z[self.yaw_pos])
        z = z[:self.dim_z]
        self.filter.update(z)
        if self.yaw_pos > -1:
            self.filter.x[self.yaw_pos] = angle_in_range(self.filter.x[self.yaw_pos])

class UKFTracker(Tracker):
    def __init__(self, z, dim_x, dim_z, dim_out, yaw_pos=2, model=None, tracking_name='car', dt=1.0):
        super().__init__(dim_x, dim_z, dim_out, yaw_pos, model)
        # sp = MerweScaledSigmaPoints(n=self.model.dim_x, alpha=0.001, beta=2.0, kappa=0.0)
        sp = JulierSigmaPoints(n=self.dim_x, kappa=0.0)
        if self.yaw_pos > -1:
            z[self.yaw_pos] = angle_in_range(z[self.yaw_pos])
        if self.dim_z == self.dim_out: # head inclusion
            self.filter = UnscentedKalmanFilter(dim_x=self.dim_x, dim_z=self.dim_z, dt=dt, fx=self.model['fx'],
                                                hx=self.model['hx'], residual_x=self.model['residual'],
                                                residual_z=self.model['residual'], points=sp)
        else:
            self.filter = UnscentedKalmanFilter(dim_x=self.dim_x, dim_z=self.dim_z, dt=dt, fx=self.model['fx'],
                                                hx=self.model['hx'], residual_x=self.model['residual'], points=sp)
        self.filter.P = model['P']
        self.filter.Q = model['Q']
        self.filter.R = model['R']
        self.filter.x[:self.dim_out] = z

    def predict(self, dt=1.0):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.filter.predict(dt)
        if self.yaw_pos > -1:
            self.filter.x[self.yaw_pos] = angle_in_range(self.filter.x[self.yaw_pos])
        self.history.append(self.filter.x)

        sigmas_h = []
        for s in self.filter.sigmas_f:
            sigmas_h.append(self.filter.hx(s))
        sigmas_h = np.atleast_2d(sigmas_h)
        if self.dim_z < self.dim_out:
            pred_z, S = unscented_transform(sigmas=sigmas_h, Wm=self.filter.Wm, Wc=self.filter.Wc,
                                            noise_cov=self.filter.R)
            out_pred_z = np.zeros((self.dim_out))
            out_S = np.zeros((self.dim_out, self.dim_out))
            out_pred_z[:self.dim_z] = pred_z
            out_S[:self.dim_z, :self.dim_z] = S
            return out_pred_z.reshape((-1, 1)), out_S
        else:
            pred_z, S = unscented_transform(sigmas=sigmas_h, Wm=self.filter.Wm, Wc=self.filter.Wc,
                                            noise_cov=self.filter.R, residual_fn=self.filter.residual_z)
            if self.yaw_pos > -1:
                pred_z[self.yaw_pos] = angle_in_range(pred_z[self.yaw_pos])
            return pred_z.reshape((-1, 1)), S # self.filter.x # self.history[-1]

    def update(self, z):
        self.history = []
        if self.yaw_pos > -1: # head inclusion
            z[self.yaw_pos] = angle_in_range(z[self.yaw_pos])
        z = z[:self.dim_z]
        self.filter.update(z)
        if self.yaw_pos > -1:
            self.filter.x[self.yaw_pos] = angle_in_range(self.filter.x[self.yaw_pos])


class IMMTracker(Tracker):
    def __init__(self, z, dim_x, dim_z, dim_out, yaw_pos=2, model=None, tracking_name='car', dt=1.0):
        super().__init__(dim_x, dim_z, dim_out, yaw_pos, model)
        filters = []
        if self.yaw_pos > -1:
            z[self.yaw_pos] = angle_in_range(z[self.yaw_pos])
        mode_probs = np.ones(len(model))
        transition_prob_mtx = np.eye(len(model))
        for idx, m in enumerate(model):
            # sp = MerweScaledSigmaPoints(n=self.dim_x, alpha=0.001, beta=2.0, kappa=0.0)
            sp = JulierSigmaPoints(n=self.dim_x, kappa=0.0)
            ftr = UnscentedKalmanFilter(dim_x=self.dim_x, dim_z= self.dim_z, dt=dt, fx=m['fx'], hx=m['hx'], residual_x=m['residual'], residual_z=m['residual'], points=sp)
            ftr.P = m['P']
            ftr.Q = m['Q']
            ftr.R = m['R']
            ftr.x[:self.dim_out] = z
            filters.append(ftr)
            mode_probs[idx] = m['mode_prob']
            transition_prob_mtx[idx, :] = (1.0 - m['transition_prob'])/(len(model) - 1.0)
            transition_prob_mtx[idx, idx] = m['transition_prob']

        mode_probs = mode_probs / np.sum(mode_probs)
        self.filter = IMMEstimator(filters, mode_probs, transition_prob_mtx)

    def predict(self, dt=1.0):
        self.filter.predict(dt)
        if self.yaw_pos > -1:
            self.filter.x[self.yaw_pos] = angle_in_range(self.filter.x[self.yaw_pos])
        self.history.append(self.filter.x)

        # max det filter?
        det_Ss = []
        Ss = []
        pred_zs = []
        for idx, ftr in enumerate(self.filter.filters):
            sigmas_h = []
            for s in ftr.sigmas_f:
                sigmas_h.append(ftr.hx(s))
            sigmas_h = np.atleast_2d(sigmas_h)
            pred_z, S = unscented_transform(sigmas=sigmas_h, Wm=ftr.Wm, Wc=ftr.Wc, noise_cov=ftr.R, residual_fn=ftr.residual_z)
            det_Ss.append(np.linalg.det(S))
            Ss.append(S)
            pred_zs.append(pred_z)

        max_arg = np.argmax(np.array(det_Ss))
        max_S = Ss[max_arg]
        max_pred_z = pred_zs[max_arg]
        if self.yaw_pos > -1:
            max_pred_z[self.yaw_pos] = angle_in_range(max_pred_z[self.yaw_pos])

        return max_pred_z.reshape((-1, 1)), max_S

    def update(self, z):
        self.history = []
        if self.yaw_pos > -1:
            z[self.yaw_pos] = angle_in_range(z[self.yaw_pos])
        self.filter.update(z)
        if self.yaw_pos > -1:
            self.filter.x[self.yaw_pos] = angle_in_range(self.filter.x[self.yaw_pos])



# class KalmanBoxTracker(object):
#     """
#     This class represents the internel state of individual tracked objects observed as bbox.
#     """
#     count = 0
#
#     def __init__(self, bbox3D, info, covariance_id=0, track_score=None, tracking_name='car',
#                  use_angular_velocity=False):
#         """
#         Initialises a tracker using initial bounding box.
#         """
#         # define constant velocity model
#         if not use_angular_velocity:
#             self.kf = KalmanFilter(dim_x=10, dim_z=7)
#             self.kf.F = np.array([[1, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # state transition matrix
#                                   [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
#                                   [0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
#                                   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#                                   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#                                   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#                                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#                                   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#                                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
#
#             self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # measurement function,
#                                   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#                                   [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#                                   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#                                   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#                                   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#                                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]])
#         else:
#             # with angular velocity
#             self.kf = KalmanFilter(dim_x=11, dim_z=7)
#             self.kf.F = np.array([[1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # state transition matrix
#                                   [0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#                                   [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
#                                   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
#                                   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#                                   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#                                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
#                                   [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
#                                   [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
#                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
#                                   [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
#
#             self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # measurement function,
#                                   [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#                                   [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
#                                   [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
#                                   [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
#                                   [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
#                                   [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
#
#         # Initialize the covariance matrix, see covariance.py for more details
#         if covariance_id == 0:  # exactly the same as AB3DMOT baseline
#             # self.kf.R[0:,0:] *= 10.   # measurement uncertainty
#             self.kf.P[7:,
#             7:] *= 1000.  # state uncertainty, give high uncertainty to the unobservable initial velocities, covariance matrix
#             self.kf.P *= 10.
#
#             # self.kf.Q[-1,-1] *= 0.01    # process uncertainty
#             self.kf.Q[7:, 7:] *= 0.01
#         elif covariance_id == 1:  # for kitti car, not supported
#             covariance = Covariance(covariance_id)
#             self.kf.P = covariance.P
#             self.kf.Q = covariance.Q
#             self.kf.R = covariance.R
#         elif covariance_id == 2:  # for nuscenes
#             covariance = Covariance(covariance_id)
#             self.kf.P = covariance.P[tracking_name]
#             self.kf.Q = covariance.Q[tracking_name]
#             self.kf.R = covariance.R[tracking_name]
#             if not use_angular_velocity:
#                 self.kf.P = self.kf.P[:-1, :-1]
#                 self.kf.Q = self.kf.Q[:-1, :-1]
#         else:
#             assert (False)
#
#         self.kf.x[:7] = bbox3D.reshape((7, 1))
#
#         self.time_since_update = 0
#         self.id = KalmanBoxTracker.count
#         KalmanBoxTracker.count += 1
#         self.history = []
#         self.hits = 1  # number of total hits including the first detection
#         self.hit_streak = 1  # number of continuing hit considering the first detection
#         self.first_continuing_hit = 1
#         self.still_first = True
#         self.age = 0
#         self.info = info  # other info
#         self.track_score = track_score
#         self.tracking_name = tracking_name
#         self.use_angular_velocity = use_angular_velocity
#
#     def update(self, bbox3D, info):
#         """
#         Updates the state vector with observed bbox.
#         """
#         self.time_since_update = 0
#         self.history = []
#         self.hits += 1
#         self.hit_streak += 1  # number of continuing hit
#         if self.still_first:
#             self.first_continuing_hit += 1  # number of continuing hit in the fist time
#
#         ######################### orientation correction
#         if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2  # make the theta still in the range
#         if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
#
#         new_theta = bbox3D[3]
#         if new_theta >= np.pi: new_theta -= np.pi * 2  # make the theta still in the range
#         if new_theta < -np.pi: new_theta += np.pi * 2
#         bbox3D[3] = new_theta
#
#         predicted_theta = self.kf.x[3]
#         if abs(new_theta - predicted_theta) > np.pi / 2.0 and abs(
#                 new_theta - predicted_theta) < np.pi * 3 / 2.0:  # if the angle of two theta is not acute angle
#             self.kf.x[3] += np.pi
#             if self.kf.x[3] > np.pi: self.kf.x[3] -= np.pi * 2  # make the theta still in the range
#             if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
#
#         # now the angle is acute: < 90 or > 270, convert the case of > 270 to < 90
#         if abs(new_theta - self.kf.x[3]) >= np.pi * 3 / 2.0:
#             if new_theta > 0:
#                 self.kf.x[3] += np.pi * 2
#             else:
#                 self.kf.x[3] -= np.pi * 2
#
#         #########################
#
#         self.kf.update(bbox3D)
#
#         if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2  # make the theta still in the range
#         if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
#         self.info = info
#
#     def predict(self):
#         """
#         Advances the state vector and returns the predicted bounding box estimate.
#         """
#         self.kf.predict()
#         if self.kf.x[3] >= np.pi: self.kf.x[3] -= np.pi * 2
#         if self.kf.x[3] < -np.pi: self.kf.x[3] += np.pi * 2
#
#         self.age += 1
#         if (self.time_since_update > 0):
#             self.hit_streak = 0
#             self.still_first = False
#         self.time_since_update += 1
#         self.history.append(self.kf.x)
#         return self.history[-1]
#
#     def get_state(self):
#         """
#         Returns the current bounding box estimate.
#         """
#         return self.kf.x[:7].reshape((7,))



        # motion tracker
        # if error yaw is zero mean && error yaw is small variance
        # headed ukf: x, y, yaw, v, (yaw')
        # if not
        # not headed kf: x, y, x', y'
        # zwlh # zero mean inspection [mean_z, mean_w, mean_l, mean_h]
        # std dev inspection [sd_z, sd_w, sd_l, sd_h]
        # covariance = Covariance(2)
        # self.P = covariance.P #[tracking_name]
        # self.Q = covariance.Q #[tracking_name]
        # self.R = covariance.R #[tracking_name]
        # mean size (w, l, h), size error (dw, dl, dh), position error (dx, dy, dz)
        # mean velocity (x', y', z', yaw'), velocity error (dx', dy', dz', dyaw'),
        # class-wise or value-wise (mean, error)
        # holonomic or non-holonimic

        # P and R from detection error
        # Q from ground truth motion change

        # option
        # motion models: rm, cv, ctrv, ctcv, ctpv
            # small heading change error: holonomic
            #
        # covariance: dx, dy, dyaw, dz, dl, dw, dh, dvx, dvy, dvyaw, dvz
        # filter: kf, ukf, imm
        # association probability: iou (with augmentation factor), mahalanobis,
            # distance factor: x, y, yaw, z, l, w, h
        # limitation: angular velocity
            # how to apply angular velocity limitation
        # data association: greedy, hungarian, jpda

        # cov = Covariance('nuscenes', 'train')
        # self.measurement_noise_mean = cov.measurement_noise_mean
        # self.process_noise_mean = cov.process_noise_mean
        # self.measurement_noise_std_dev = cov.measurement_noise_std_dev
        # self.process_noise_std_dev = cov.process_noise_std_dev
        # for tracking_name in NUSCENES_TRACKING_NAMES:
        #     # not predictable => value should be zero mean
        #     pnm_mask = np.less(self.process_noise_mean[tracking_name] < cov.process_noise_mean_trigger)
        #     pnsd_mask = np.less(self.process_noise_std_dev[tracking_name] < cov.process_noise_std_dev_trigger)
        #     # not detectable => value should be zero mean
        #     mnm_mask = np.less(self.measurement_noise_mean[tracking_name], cov.measurement_noise_mean_trigger)
        #     mnsd_mask = np.less(self.measurement_noise_std_dev[tracking_name] < cov.measurement_noise_std_dev_trigger)
        #
        #     pn_mask[tracking_name] = np.logical_and(pnm_mask, pnsd_mask) # not predictable ? controllability (complete state controllability)
        #     mn_mask[tracking_name] = np.logical_and(mnsd_mask, mnm_mask) # not detectable ? observability?
        #
        #     if pn_mask[yaw_pos]:
        #         self.head_motion_on[tracking_name] = True
        #     if mn_mask[yaw_pos]:
        #         self.head_appearance_on[tracking_name] = True
        #
        #     if self.head_motion_on[tracking_name] and self.head_appearance_on[tracking_name]:



        # if not self.head_motion_on and not self.head_appearance_on[tracking_name]:
        #     mstate_on = np.array([True, True, True])
        #     num_mstate = 3
        #     m_z = m_z[0:num_mstate]
        #     mtracker = KFTracker(z=m_z, dim_x=5, dim_z=3, tracking_name=tracking_name)
        #
        # elif not self.head_appearance_on[tracking_name]:
        #
        #
        # elif not self.head_motion_on[tracking_name]:
        #
        # else:
        #     mstate_on = np.array([True, True, True])
        #     num_mstate = 3
        #     m_z = m_z[0:num_mstate]
        #     if not self.imm_on: # heading variance? heading diff variance?
        #         model = {}
        #         model['fx'] = mm.cv_fx # cv vs. ctrv
        #         model['hx'] = mm.hx
        #         model['residual_fn'] = mm.residual
        #         model['P'] = cov.mP[tracking_name]
        #         model['Q'] = cov.mP[tracking_name]
        #         model['R'] = cov.mP[tracking_name]
        #         mtracker = UKFTracker(z=m_z, dim_x=5, dim_z=3, yaw_pos=2, model=model, tracking_name=tracking_name)
        #     else:
                # models = []
                # # cv
                # model['fx'] = mm.cv_fx
                # model['hx'] = mm.hx
                # model['residual_fn'] = mm.residual
                # model['mode_prob'] = 0.3
                # model['transition_prob'] = 0.96
                # model['P'] = cov.mP[tracking_name]
                # model['Q'] = cov.mP[tracking_name]
                # model['R'] = cov.mP[tracking_name]
                # models.append(model)
                # # ctrv
                # model['fx'] = mm.cv_fx
                # model['hx'] = mm.hx
                # model['residual_fn'] = mm.residual
                # model['mode_prob'] = 0.3
                # model['transition_prob'] = 0.96
                # model['P'] = cov.mP[tracking_name]
                # model['Q'] = cov.mP[tracking_name]
                # model['R'] = cov.mP[tracking_name]
                # models.append(model)
                # # rm
                # model['fx'] = mm.cv_fx
                # model['hx'] = mm.hx
                # model['residual_fn'] = mm.residual
                # model['mode_prob'] = 0.3
                # model['transition_prob'] = 0.96
                # model['P'] = cov.mP[tracking_name]
                # model['Q'] = cov.mP[tracking_name]
                # model['R'] = cov.mP[tracking_name]
                # models.append(model)
                # mtracker = IMMTracker(z=m_z, dim_x=5, dim_z=3, tracking_name=tracking_name)


