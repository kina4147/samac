import numpy as np
from filterpy.kalman import unscented_transform
from filterpy.kalman import (KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter)
from filterpy.kalman import (unscented_transform, MerweScaledSigmaPoints, JulierSigmaPoints)
from filterpy.kalman import IMMEstimator
from .motion_models import cv_fx, ctrv_fx, rm_fx, hx_2, hx_3, residual
from .params import Covariance
from .utils import angle_in_range

from collections import namedtuple
SAMACTracker = namedtuple('SAMACTracker', ['info', 'mtracker', 'atracker'])
class TrackerGenerator(object):
    def __init__(self, covariance_id=0, tracking_name='car'):
        self.tracking_name = tracking_name
        # x, y, z, yaw, w, l, h, x', y', z', yaw', w', l', h'
        self.association_metrics = ['mdist'] # 'iou' or 'mdist' 'mahalanobis distance'

        self.m_state_on = np.array([True, True, True])
        self.dim_m_x = 5
        self.dim_m_z = 3

        self.a_state_on = np.array([True, False, False, True])
        self.dim_a_x = 4
        self.dim_a_z = 4

        self.imm_on = False
        self.m_yaw_pos = -1
        self.a_yaw_pos = -1
        # for tracking_name in NUSCENES_TRACKING_NAMES:
        if tracking_name == 'bus': # motional
            self.head_measure_on = True
            self.head_track_on = True
            self.association_metrics = ['mdist']
            self.da_state_on = np.array([True, True, True, False, True, True, True])
        elif tracking_name == 'car' or tracking_name=='Car': # motional
            # self.imm_on = True
            self.head_measure_on = True
            self.head_track_on = True
            self.association_metrics = ['mdist']
            self.da_state_on = np.array([True, True, True, False, True, True, True])
        elif tracking_name == 'truck':
            self.head_measure_on = True
            self.head_track_on = True
            self.association_metrics = ['mdist']
            self.da_state_on = np.array([True, True, True, False, True, True, True])
        elif tracking_name == 'motorcycle':
            # self.imm_on = True
            # self.head_measure_on = False
            # self.head_track_on = True
            # self.da_state_on = np.array([True, True, False, False, True, True, True])
            # self.association_metrics = ['mdist']
            self.head_measure_on = True
            self.head_track_on = True
            self.da_state_on = np.array([True, True, True, False, True, True, True])
            self.association_metrics = ['mdist']
        elif tracking_name == 'pedestrian' or tracking_name =='Pedestrian':
            # self.imm_on = True
            self.head_measure_on = False
            self.head_track_on = False
            self.da_state_on = np.array([True, True, False, False, False, False, True])
            self.association_metrics = ['mdist']
        elif tracking_name == 'bicycle' or tracking_name =='Cyclist': # holonomic
            # self.imm_on = True
            # self.head_measure_on = True # False
            # self.head_track_on = True
            # self.association_metrics = ['mdist']
            # self.da_state_on = np.array([True, True, True, False, True, True, True])
            self.head_measure_on = False
            self.head_track_on = False
            self.association_metrics = ['mdist']
            self.da_state_on = np.array([True, True, False, False, False, False, True])
        elif tracking_name == 'trailer':
            self.head_measure_on = False
            self.head_track_on = True
            self.association_metrics = ['mdist']
            self.da_state_on = np.array([True, True, False, False, True, False, True])
        else:
            assert False

        cov = Covariance(covariance_id, tracking_name)
        self.mmodel = {}
        if self.head_measure_on:  # x, y, theta
            self.m_state_on = np.array([True, True, True, False, False, False, False])
            self.m_yaw_pos = 2
            self.a_yaw_pos = -1
            self.dim_m_z = np.count_nonzero(self.m_state_on)
            if self.head_track_on:  # x, y, theta, v, theta'
                self.linear_motion = False
                self.dim_m_x = 5 # x, y, theta, v, theta'
                self.mmodel['fx'] = cv_fx # cv vs. ctrv
                self.mmodel['hx'] = hx_3
                self.mmodel['residual'] = residual
                self.mmodel['P'] = cov.m_head_P[:self.dim_m_x, :self.dim_m_x]
                self.mmodel['Q'] = cov.m_head_Q[:self.dim_m_x, :self.dim_m_x]
                self.mmodel['R'] = cov.m_R[:self.dim_m_z, :self.dim_m_z]
            else:
                self.linear_motion = True
                self.dim_m_x = 6 # x, y, theta, x', y', theta'
                self.mmodel['P'] = cov.m_no_head_P[:self.dim_m_x, :self.dim_m_x]
                self.mmodel['Q'] = cov.m_no_head_Q[:self.dim_m_x, :self.dim_m_x]
                self.mmodel['R'] = cov.m_R[:self.dim_m_z, :self.dim_m_z]

            # z, w, l, h
            self.dim_a_x = 4
            self.dim_a_z = 4
            self.a_state_on = np.array([False, False, False, True, True, True, True])
            self.amodel = {}
            self.amodel['P'] = cov.a_P[1:self.dim_a_x+1, 1:self.dim_a_x+1]
            self.amodel['Q'] = cov.a_Q[1:self.dim_a_x+1, 1:self.dim_a_x+1]
            self.amodel['R'] = cov.a_R[1:self.dim_a_z+1, 1:self.dim_a_z+1]

        else: # x, y, but initialized by measure
            self.m_state_on = np.array([True, True, False, False, False, False, False])
            self.a_yaw_pos = 0
            self.dim_m_x = 4 # x, y, theta, v # x, y, x', y'
            self.dim_m_z = 2 # x, y, theta, v # x, y, x', y'
            if self.head_track_on:  # x, y, theta, v
                self.linear_motion = False
                self.m_yaw_pos = 2
                self.mmodel['fx'] = cv_fx # cv vs. ctrv
                self.mmodel['hx'] = hx_2
                self.mmodel['residual'] = residual
                self.mmodel['P'] = cov.m_head_P[:self.dim_m_x, :self.dim_m_x]
                self.mmodel['Q'] = cov.m_head_Q[:self.dim_m_x, :self.dim_m_x]
                self.mmodel['R'] = cov.m_R[:self.dim_m_z, :self.dim_m_z]
            else:
                self.linear_motion = True
                self.m_yaw_pos = -1
                self.mmodel['P'] = cov.m_no_head_P[:self.dim_m_x, :self.dim_m_x]
                self.mmodel['Q'] = cov.m_no_head_Q[:self.dim_m_x, :self.dim_m_x]
                self.mmodel['R'] = cov.m_R[:self.dim_m_z, :self.dim_m_z]

            # theta, z, w, l, h
            self.dim_a_x = 5
            self.dim_a_z = 5
            self.a_state_on = np.array([False, False, True, True, True, True, True])
            # self.a_da_state_on = self.a_da_state_on[-self.dim_a_z-1:-1]
            self.amodel = {}
            self.amodel['P'] = cov.a_P[0:self.dim_a_x, 0:self.dim_a_x]
            self.amodel['Q'] = cov.a_Q[0:self.dim_a_x, 0:self.dim_a_x]
            self.amodel['R'] = cov.a_R[0:self.dim_a_z, 0:self.dim_a_z]

        # self.state_on = np.concatenate((self.m_state_on, self.a_state_on))
        # self.state_cov_on = self.state_on.reshape(-1, 1).dot(self.state_on.reshape(-1, 1).T)
        # self.dim_z = np.count_nonzero(self.state_on)

    def generate_tracker(self, z, info):
        # z: x, y, yaw, z, w, l, h
        a_z = np.copy(z[self.a_state_on])
        if self.m_yaw_pos > -1:
            init_m_state_on = np.copy(self.m_state_on)
            init_m_state_on[self.m_yaw_pos] = True
            m_z = np.copy(z[init_m_state_on])
        else:
            m_z = np.copy(z[self.m_state_on])


        # Motion Tracker Initialization
        if self.linear_motion: # x, y, theta, x', y', theta' or x, y, theta, x', y'
            m_tracker = KFTracker(z=m_z, dim_x=self.dim_m_x, dim_z=self.dim_m_z, yaw_pos=self.m_yaw_pos, model=self.mmodel, tracking_name=self.tracking_name)
        else:
            if not self.imm_on:
                m_tracker = UKFTracker(z=m_z, dim_x=self.dim_m_x, dim_z=self.dim_m_z, yaw_pos=self.m_yaw_pos, model=self.mmodel, tracking_name=self.tracking_name)
            else:
                models = []
                # # cv_fx
                model_1 = {'fx': cv_fx, 'hx': self.mmodel['hx'], 'residual': self.mmodel['residual'], 'mode_prob': 0.3, 'transition_prob': 0.96, 'P': self.mmodel['P'], 'Q': self.mmodel['Q'], 'R':self.mmodel['R']}
                models.append(model_1)
                # # ctrv_fx
                # model_2 = {'fx': ctrv_fx, 'hx': self.mmodel['hx'], 'residual': self.mmodel['residual'], 'mode_prob': 0.3, 'transition_prob': 0.96, 'P': self.mmodel['P'], 'Q': self.mmodel['Q'], 'R':self.mmodel['R']}
                # models.append(model_2)
                # # rm_fx
                model_3 = {'fx': rm_fx, 'hx': self.mmodel['hx'], 'residual': self.mmodel['residual'], 'mode_prob': 0.3, 'transition_prob': 0.96, 'P': 1.5*self.mmodel['P'], 'Q': 1.5*self.mmodel['Q'], 'R':1.5*self.mmodel['R']}
                models.append(model_3)
                m_tracker = IMMTracker(z=m_z, dim_x=self.dim_m_x, dim_z=self.dim_m_z, yaw_pos=self.m_yaw_pos, model=models, tracking_name=self.tracking_name)


        # Appearance Tracker Initialization
        a_tracker = KFTracker(z=a_z, dim_x=self.dim_a_x, dim_z=self.dim_a_z, yaw_pos=self.a_yaw_pos, model=self.amodel, tracking_name=self.tracking_name)

        # Track Info Initialization
        tracker_info = TrackerInfo(m_state_on=self.m_state_on, a_state_on=self.a_state_on, info=info, tracking_name=self.tracking_name)
        tracker = SAMACTracker(tracker_info, m_tracker, a_tracker)
        return tracker



class TrackerInfo(object):
    count = 0
    def __init__(self, m_state_on=np.ones(7, dtype=bool), a_state_on=np.ones(7, dtype=bool), info=None, tracking_name='car'):
        self.id = TrackerInfo.count
        TrackerInfo.count += 1
        self.time_since_update = 0
        self.hits = 1  # number of total hits including the first detection
        self.hit_streak = 0  # number of continuing hit considering the first detection
        self.continue_hits = 0
        self.age = 0
        self.info = info
        self.tracking_name = tracking_name
        self.m_state_on = m_state_on
        self.a_state_on = a_state_on
        # self.state_cov_on = self.state_on.reshape(-1, 1).dot(self.state_on.reshape(-1, 1).T)
        self.dist = 0
        self.wscore = 0
        self.tracking = False
        self.m_prior = 0.8
        self.a_prior = 0.4
        self.m_likelihood = 0.5
        self.a_likelihood = 0.5

    def predict(self):
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
            self.still_first = False
        else:
            self.continue_hits = self.hit_streak
        self.time_since_update += 1

    def update(self, info):
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1  # number of continuing hit
        self.continue_hits = self.hit_streak
        self.info = info


class Tracker(object):
    def __init__(self, dim_x, dim_z, yaw_pos, model=None):
        self.dim_x = dim_x
        self.dim_z = dim_z
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
        return self.filter.x[:self.dim_z].reshape((-1, 1))

class KFTracker(Tracker): # x, y, x', y'
    def __init__(self, z, dim_x, dim_z, yaw_pos=2, model=None, tracking_name='car'):
        super().__init__(dim_x, dim_z, yaw_pos, model)
        self.filter = KalmanFilter(dim_x=self.dim_x, dim_z=self.dim_z)
        self.filter.P = model['P']
        self.filter.Q = model['Q']
        self.filter.R = model['R']
        self.filter.F = np.eye(self.dim_x) + np.eye(self.dim_x, k=self.dim_z)
        self.filter.H = np.eye(self.dim_z, self.dim_x)
        self.filter.x[:len(z), :] = z.reshape((-1, 1))

        if self.yaw_pos > -1: # head inclusion
            self.filter.x[self.yaw_pos] = angle_in_range(self.filter.x[self.yaw_pos])

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
        return pred_z, S

    def update(self, z):
        self.history = []
        # z = z[:self.dim_z]
        # if self.yaw_pos > -1: # head inclusion
        #     z[self.yaw_pos] = angle_in_range(z[self.yaw_pos])
        self.filter.update(z)
        if self.yaw_pos > -1:
            self.filter.x[self.yaw_pos] = angle_in_range(self.filter.x[self.yaw_pos])


class UKFTracker(Tracker):
    def __init__(self, z, dim_x, dim_z, yaw_pos=2, model=None, tracking_name='car', dt=1.0):
        super().__init__(dim_x, dim_z, yaw_pos, model)
        # sp = MerweScaledSigmaPoints(n=self.model.dim_x, alpha=0.001, beta=2.0, kappa=0.0)
        sp = JulierSigmaPoints(n=self.dim_x, kappa=0.0)
        # if self.yaw_pos > -1:
        #     z[self.yaw_pos] = angle_in_range(z[self.yaw_pos])
        # if self.dim_z == self.dim_out: # head inclusion
        if -1 < self.yaw_pos < self.dim_z:
            self.filter = UnscentedKalmanFilter(dim_x=self.dim_x, dim_z=self.dim_z, dt=dt, fx=self.model['fx'],
                                                hx=self.model['hx'], residual_x=self.model['residual'],
                                                residual_z=self.model['residual'], points=sp)
        else:
            self.filter = UnscentedKalmanFilter(dim_x=self.dim_x, dim_z=self.dim_z, dt=dt, fx=self.model['fx'],
                                                hx=self.model['hx'], residual_x=self.model['residual'], points=sp)
        self.filter.P = model['P']
        self.filter.Q = model['Q']
        self.filter.R = model['R']
        self.filter.x[:len(z)] = z

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
        if -1 < self.yaw_pos < self.dim_z:
            pred_z, S = unscented_transform(sigmas=sigmas_h, Wm=self.filter.Wm, Wc=self.filter.Wc,
                                            noise_cov=self.filter.R, residual_fn=self.filter.residual_z)
            pred_z[self.yaw_pos] = angle_in_range(pred_z[self.yaw_pos])
        else:
            pred_z, S = unscented_transform(sigmas=sigmas_h, Wm=self.filter.Wm, Wc=self.filter.Wc,
                                            noise_cov=self.filter.R)
        return pred_z.reshape((-1, 1)), S

    def update(self, z):
        self.history = []
        # if self.yaw_pos > -1: # head inclusion
        #     z[self.yaw_pos] = angle_in_range(z[self.yaw_pos])
        # z = z[:self.dim_z]
        self.filter.update(z)
        if self.yaw_pos > -1:
            self.filter.x[self.yaw_pos] = angle_in_range(self.filter.x[self.yaw_pos])


class IMMTracker(Tracker):
    def __init__(self, z, dim_x, dim_z, yaw_pos=2, model=None, tracking_name='car', dt=1.0):
        super().__init__(dim_x, dim_z, yaw_pos, model)
        filters = []
        # if self.yaw_pos > -1:
        #     z[self.yaw_pos] = angle_in_range(z[self.yaw_pos])
        mode_probs = np.ones(len(model))
        transition_prob_mtx = np.eye(len(model))
        for idx, m in enumerate(model):
            # sp = MerweScaledSigmaPoints(n=self.dim_x, alpha=0.001, beta=2.0, kappa=0.0)
            sp = JulierSigmaPoints(n=self.dim_x, kappa=0.0)
            if -1 < self.yaw_pos < self.dim_z:
                ftr = UnscentedKalmanFilter(dim_x=self.dim_x, dim_z= self.dim_z, dt=dt, fx=m['fx'], hx=m['hx'], residual_x=m['residual'], residual_z=m['residual'], points=sp)
            else:
                ftr = UnscentedKalmanFilter(dim_x=self.dim_x, dim_z=self.dim_z, dt=dt, fx=m['fx'], hx=m['hx'],
                                            residual_x=m['residual'], points=sp)
            ftr.P = m['P']
            ftr.Q = m['Q']
            ftr.R = m['R']
            ftr.x[:len(z)] = z
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
        det_Ss = []
        Ss = []
        pred_zs = []
        for idx, ftr in enumerate(self.filter.filters):
            sigmas_h = []
            for s in ftr.sigmas_f:
                sigmas_h.append(ftr.hx(s))
            sigmas_h = np.atleast_2d(sigmas_h)
            if -1 < self.yaw_pos < self.dim_z:
                pred_z, S = unscented_transform(sigmas=sigmas_h, Wm=ftr.Wm, Wc=ftr.Wc, noise_cov=ftr.R, residual_fn=ftr.residual_z)
            else:
                pred_z, S = unscented_transform(sigmas=sigmas_h, Wm=ftr.Wm, Wc=ftr.Wc, noise_cov=ftr.R)

            det_Ss.append(np.linalg.det(S))
            Ss.append(S)
            pred_zs.append(pred_z)

        max_arg = np.argmax(np.array(det_Ss))
        max_S = Ss[max_arg]
        max_pred_z = pred_zs[max_arg]

        if -1 < self.yaw_pos < self.dim_z:
            max_pred_z[self.yaw_pos] = angle_in_range(max_pred_z[self.yaw_pos])

        return max_pred_z.reshape((-1, 1)), max_S

    def update(self, z):
        self.history = []
        # if self.yaw_pos > -1:
        #     z[self.yaw_pos] = angle_in_range(z[self.yaw_pos])
        # z = z[:self.dim_z]
        self.filter.update(z)
        if self.yaw_pos > -1:
            self.filter.x[self.yaw_pos] = angle_in_range(self.filter.x[self.yaw_pos])

