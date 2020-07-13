import numpy as np
from filterpy.kalman import unscented_transform
from filterpy.kalman import (KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter)
from filterpy.kalman import (unscented_transform, MerweScaledSigmaPoints, JulierSigmaPoints)
from filterpy.kalman import IMMEstimator
import motion_models as mm
from covariance import Covariance
from utils import angle_in_range

class TrackerLoader(object):
    def __init__(self, tracking_name='car'):
        covariance = Covariance(2)
        self.P = covariance.P #[tracking_name]
        self.Q = covariance.Q #[tracking_name]
        self.R = covariance.R #[tracking_name]
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

        if tracking_name == 'bicycle': # holonomic
            pass
        elif tracking_name == 'bus':
            pass
        elif tracking_name == 'car':
            pass
        elif tracking_name == 'motorcycle':
            pass
        elif tracking_name == 'pedestrian':
            pass
        elif tracking_name == 'trailer':
            pass
        elif tracking_name == 'truck':
            pass
        else:
            assert False
        self.tracker = None

    def get_tracker(self):
        return self.tracker



class TrackerInfo(object):
    count = 0

    def __init__(self, info, track_score=None, tracking_name='car'):
        self.id = TrackerInfo.count
        TrackerInfo.count += 1
        self.time_since_update = 0
        self.hits = 1  # number of total hits including the first detection
        self.hit_streak = 1  # number of continuing hit considering the first detection
        self.first_continuing_hit = 1
        self.still_first = True
        self.age = 0
        self.info = info  # other info
        self.track_score = track_score
        self.tracking_name = tracking_name

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
    def __init__(self, dim_x, dim_z, yaw_pos, model=None):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.yaw_pos = yaw_pos
        self.model = model
        self.yaw_pos = 2
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

class KFTracker(Tracker):
    def __init__(self, z, dim_x, dim_z, yaw_pos=2, model=None, tracking_name='car'):
        super().__init__(dim_x, dim_z, yaw_pos, model)
        self.filter = KalmanFilter(dim_x=self.dim_x, dim_z=self.dim_z)
        covariance = Covariance(3)
        self.filter.P = covariance.aP[tracking_name]
        self.filter.Q = covariance.aQ[tracking_name]
        self.filter.R = covariance.aR[tracking_name]
        self.filter.x[:self.dim_z] = z.reshape((self.dim_z, 1))
        self.filter.F = np.eye(self.dim_x) + np.eye(self.dim_x, k=self.dim_z)
        self.filter.H = np.eye(self.dim_z, self.dim_x)

    def predict(self, dt=1.0):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.filter.predict(dt)
        self.filter.x[self.yaw_pos] = angle_in_range(self.filter.x[self.yaw_pos])

        PHT = np.dot(self.filter.P, self.filter.H.T)
        S = np.dot(self.filter.H, PHT) + self.filter.R
        pred_z = np.dot(self.filter.H, self.filter.x)
        pred_z[self.yaw_pos] = angle_in_range(pred_z[self.yaw_pos])
        self.history.append(self.filter.x)
        return pred_z, S # self.filter.x # self.history[-1]

    def update(self, z):
        self.history = []
        z[self.yaw_pos] = angle_in_range(z[self.yaw_pos])
        self.filter.update(z)
        self.filter.x[self.yaw_pos] = angle_in_range(self.filter.x[self.yaw_pos])
        # dyaw = z[self.yaw_pos] - self.filter.x[self.yaw_pos]
        # dyaw = angle_in_range(dyaw)
        # if abs(dyaw) > np.pi / 2.0:
        #     z[self.yaw_pos] += np.pi


class UKFTracker(Tracker):

    def __init__(self, z, dim_x, dim_z, yaw_pos, model=None, tracking_name='car', dt=1.0):
        super().__init__(dim_x, dim_z, yaw_pos, model)
        # sp = MerweScaledSigmaPoints(n=self.model.dim_x, alpha=0.001, beta=2.0, kappa=0.0)
        sp = JulierSigmaPoints(n=self.dim_x, kappa=0.0)
        self.filter = UnscentedKalmanFilter(dim_x=self.dim_x, dim_z=self.dim_z, dt=dt, fx=self.model.fx, hx=self.model.hx, residual_x=self.model.residual_fn, residual_z=self.model.residual_fn, points=sp)

        covariance = Covariance(3)
        self.filter.P = covariance.mP[tracking_name]
        self.filter.Q = covariance.mQ[tracking_name]
        self.filter.R = covariance.mR[tracking_name]
        self.filter.x[:self.dim_z] = z

    def predict(self, dt=1.0):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.filter.predict(dt)
        self.filter.x[self.yaw_pos] = angle_in_range(self.filter.x[self.yaw_pos])

        sigmas_h = []
        for s in self.filter.sigmas_f:
            sigmas_h.append(self.filter.hx(s))
        sigmas_h = np.atleast_2d(sigmas_h)
        pred_z, S = unscented_transform(sigmas=sigmas_h, Wm=self.filter.Wm, Wc=self.filter.Wc, noise_cov=self.filter.R, residual_fn=self.filter.residual_z)
        pred_z[self.yaw_pos] = angle_in_range(pred_z[self.yaw_pos])
        self.history.append(self.filter.x)
        return pred_z.reshape((-1, 1)), S # self.filter.x # self.history[-1]

    def update(self, z):
        self.history = []
        z[self.yaw_pos] = angle_in_range(z[self.yaw_pos])
        self.filter.update(z)
        self.filter.x[self.yaw_pos] = angle_in_range(self.filter.x[self.yaw_pos])

        # dyaw = z[self.yaw_pos] - self.filter.x[self.yaw_pos]
        # dyaw = angle_in_range(dyaw)
        # if abs(np.pi - abs(dyaw)) <= np.pi / 6.0: # 30 deg (
        #     z[self.yaw_pos] += np.pi
        # z[self.yaw_pos] = angle_in_range(z[self.yaw_pos])


class IMMTracker(Tracker):
    def __init__(self, z, dim_x, dim_z, yaw_pos=2, model=None, tracking_name='car', dt=1.0):
        super().__init__(dim_x, dim_z, yaw_pos, model)
        filters = []
        mode_probs = np.ones(len(model))
        transition_prob_mtx = np.eye(len(model))
        for idx, m in enumerate(model):
            sp = JulierSigmaPoints(n=self.dim_x, kappa=0.0)
            # sp = MerweScaledSigmaPoints(n=self.dim_x, alpha=0.001, beta=2.0, kappa=0.0)
            ftr = UnscentedKalmanFilter(dim_x=self.dim_x, dim_z= self.dim_z, dt=dt, fx=m.fx, hx=m.hx, residual_x=m.residual_fn, residual_z=m.residual_fn, points=sp)
            covariance = Covariance(3)
            ftr.P = covariance.mP[tracking_name]
            notQmask = np.logical_not(covariance.Qmask[m.model])
            Q = np.copy(covariance.mQ[tracking_name])
            Q[notQmask, notQmask] = 1e-7
            ftr.Q = covariance.Qratio[m.model] * Q
            ftr.R = covariance.mR[tracking_name]
            ftr.x[:self.dim_z] = z
            filters.append(ftr)
            mode_probs[idx] = m.mode_prob
            transition_prob_mtx[idx, :] = (1.0 - m.transition_prob)/(len(model) - 1.0)
            transition_prob_mtx[idx, idx] = m.transition_prob

        mode_probs = mode_probs / np.sum(mode_probs)
        self.filter = IMMEstimator(filters, mode_probs, transition_prob_mtx)

    def predict(self, dt=1.0):
        self.filter.predict(dt)
        self.filter.x[self.yaw_pos] = angle_in_range(self.filter.x[self.yaw_pos])

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
        max_pred_z[self.yaw_pos] = angle_in_range(max_pred_z[self.yaw_pos])

        self.history.append(self.filter.x)
        return max_pred_z.reshape((-1, 1)), max_S

    def update(self, z):
        self.history = []
        z[self.yaw_pos] = angle_in_range(z[self.yaw_pos])
        self.filter.update(z)
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