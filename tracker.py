import numpy as np
from filterpy.kalman import unscented_transform
from filterpy.kalman import (KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter)
from filterpy.kalman import (unscented_transform, MerweScaledSigmaPoints, JulierSigmaPoints)
from filterpy.kalman import IMMEstimator
import motion_models as mm
from covariance import Covariance
from utils import angle_in_range

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
    def __init__(self, model=None):
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
        return self.filter.x[:self.model.dim_z].reshape((-1, 1))

class KFTracker(Tracker):
    def __init__(self, z, model=None, tracking_name='car'):
        super().__init__(model)
        self.filter = KalmanFilter(dim_x=self.model.dim_x, dim_z=self.model.dim_z)
        covariance = Covariance(3)
        self.filter.P = covariance.aP[tracking_name]
        self.filter.Q = covariance.aQ[tracking_name]
        self.filter.R = covariance.aR[tracking_name]
        self.filter.x[:self.model.dim_z] = z.reshape((self.model.dim_z, 1))
        self.filter.F = np.eye(self.model.dim_x) + np.eye(self.model.dim_x, k=self.model.dim_z)
        self.filter.H = np.eye(self.model.dim_z, self.model.dim_x)

    def predict(self, dt=1.0):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.filter.predict(dt)
        self.filter.x[self.yaw_pos] = angle_in_range(self.filter.x[self.yaw_pos])

        PHT = np.dot(self.filter.P, self.filter.H.T)
        S = np.dot(self.filter.H, PHT) + self.filter.R
        pred_z = np.dot(self.filter.H, self.filter.x)

        self.history.append(self.filter.x)
        return pred_z, S # self.filter.x # self.history[-1]

    def update(self, z):
        self.history = []

        dyaw = z[self.yaw_pos] - self.filter.x[self.yaw_pos]
        dyaw = angle_in_range(dyaw)
        if abs(dyaw) > np.pi / 2.0:
            z[self.yaw_pos] += np.pi
            z[self.yaw_pos] = angle_in_range(z[self.yaw_pos])

        self.filter.update(z)
        self.filter.x[self.yaw_pos] = angle_in_range(self.filter.x[self.yaw_pos])



class UKFTracker(Tracker):

    def __init__(self, z, model=None, tracking_name='car', dt=1.0):
        super().__init__(model)
        # self.filter = KalmanFilter(dim_x=self.model.dim_x, dim_z=self.model.dim_z)
        # sp = MerweScaledSigmaPoints(n=self.model.dim_x, alpha=0.001, beta=2.0, kappa=0.0)
        sp = JulierSigmaPoints(n=self.model.dim_x, kappa=0.0)
        self.filter = UnscentedKalmanFilter(dim_x=self.model.dim_x, dim_z=self.model.dim_z, dt=dt, fx=self.model.fx, hx=self.model.hx, residual_x=self.model.residual_fn, residual_z=self.model.residual_fn, points=sp)

        covariance = Covariance(3)
        self.filter.P = covariance.mP[tracking_name]
        self.filter.Q = covariance.mQ[tracking_name]
        self.filter.R = covariance.mR[tracking_name]
        self.filter.x[:self.model.dim_z] = z


    def predict(self, dt=1.0):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        self.filter.predict(dt)
        self.filter.x[self.yaw_pos] = angle_in_range(self.filter.x[self.yaw_pos])

        sigmas_h = []
        for s in self.filter.sigmas_f:
            sigmas_h.append(self.model.hx(s))
        sigmas_h = np.atleast_2d(sigmas_h)
        pred_z, S = unscented_transform(sigmas=sigmas_h, Wm=self.filter.Wm, Wc=self.filter.Wc, noise_cov=self.filter.R, residual_fn=self.model.residual_fn)
        pred_z[self.yaw_pos] = angle_in_range(pred_z[self.yaw_pos])
        self.history.append(self.filter.x)
        return pred_z.reshape((-1, 1)), S # self.filter.x # self.history[-1]

    def update(self, z):
        self.history = []
        z[self.yaw_pos] = angle_in_range(z[self.yaw_pos])
        dyaw = z[self.yaw_pos] - self.filter.x[self.yaw_pos]
        dyaw = angle_in_range(dyaw)
        if abs(np.pi - abs(dyaw)) <= np.pi / 6.0: # 30 deg (
            z[self.yaw_pos] += np.pi
            z[self.yaw_pos] = angle_in_range(z[self.yaw_pos])

        self.filter.update(z)
        self.filter.x[self.yaw_pos] = angle_in_range(self.filter.x[self.yaw_pos])



class IMMTracker(Tracker):
    def __init__(self, z, models=None, tracking_name='car', dt=1.0):
        super().__init__(models)
        self.filters = []
        for m in models:
            sp = JulierSigmaPoints(n=self.model.dim_x, kappa=0.0)
            filter = UnscentedKalmanFilter(dim_x=self.model.dim_x, dim_z=self.model.dim_z, dt=dt, fx=self.model.fx, hx=self.model.hx, residual_x=self.model.residual_fn, residual_z=self.model.residual_fn, points=sp)
            covariance = Covariance(3)
            filter.P = covariance.mP[tracking_name]
            filter.Q = covariance.mQ[tracking_name]
            filter.R = covariance.mR[tracking_name]
            filter.x[:self.model.dim_z] = z
            self.filters.append(filter)



        pass

    def predict(self, dt=0.1):
        for filter in self.filters:
            filter.predict(dt)

    def update(self, bbox3D, info):
        pass


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