import numpy as np
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box
from nuscenes.eval.common.utils import quaternion_yaw
from .data_association import samac_associate
from .tracker import TrackerGenerator
from .utils import angle_in_range
from .params import MahalanobisDistThres, Covariance

class SAMAC3DMOT(object):
    def __init__(self, covariance_id=0, tracking_name='car', dataset2samac = [0,1,2,3,4,5,6], samac2dataset=[0,1,2,3,4,5,6], max_track_age=3, max_tracklet_age=2, min_hits=2, tracking_dataset='nuscenes'): #, tracking_nuscenes=False):

        self.max_track_age = max_track_age
        self.max_tracklet_age = max_tracklet_age
        self.min_hits = min_hits
        self.trackers = []
        self.frame_count = 0
        """
        observation:
          before reorder: [x, y, z, yaw, w, l, h]
          after reorder:  [x, y, yaw, z, w, l, h]
        state:
          [x, y, yaw, z, w, l, h]
        """
        self.dataset2samac = dataset2samac
        self.samac2dataset = samac2dataset

        self.covariance_id = covariance_id
        self.tracking_name = tracking_name # object_type
        # self.tracking_nuscenes = tracking_nuscenes
        self.tracker_generator = TrackerGenerator(covariance_id, self.tracking_name)
        mdist = MahalanobisDistThres()
        self.mdist_thres = mdist.mdist_threshold

    def update(self, dets_all=None):#, ego_position, ego_orientation): #pose_translation=None, pose_rotation=None, cs_translation=None, cs_rotation=None):
        """
        Params:
          dets_all: dict
            dets - a numpy array of detections in the format [[x,y,z,theta,l,w,h],[x,y,z,theta,l,w,h],...]
            info: a array of other info for each det
        Requires: this method must be called once for each frame even with empty detections.
        Returns the a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        # print(self.tracking_name, "==========================")
        yaw_pos = 2
        self.frame_count += 1
        dets, info = dets_all['dets'], dets_all['info']  # dets: N x 7, float numpy array

        # print("is b4 dets: ", dets)
        if len(dets) > 0:
            dets = dets[:, self.dataset2samac]
        # print("is dets: ", dets)
        # trks = []
        # trks_S = []
        ret = []
        prior = []
        # trks_tracking = []
        # trks_info = []
        m_trks = []
        a_trks = []
        m_trks_S = []
        a_trks_S = []
        # PREDICTION
        for t, trk in enumerate(self.trackers):
            trk.info.predict()
            m_pred_z, m_S = trk.mtracker.predict()
            a_pred_z, a_S = trk.atracker.predict()
            # print(m_pred_z.shape, a_pred_z.shape, m_S.shape, a_S.shape)
            m_trks.append(m_pred_z)
            a_trks.append(a_pred_z)
            m_trks_S.append(m_S)
            a_trks_S.append(a_S)
            pred_z = np.zeros((7, 1))
            pred_z[self.tracker_generator.m_state_on] = m_pred_z
            pred_z[self.tracker_generator.a_state_on] = a_pred_z
            # wscore, dist, detection_score######################### more info
            # tbox = Box(center=[pred_z[0], pred_z[1], pred_z[3]], size=[pred_z[4], pred_z[5], pred_z[6]],
            #            orientation=Quaternion(axis=[0., 0., 1.], angle=pred_z[2]))
            # tbox.translate(pose_translation)
            # tbox.rotate(pose_rotation)
            # tbox.translate(cs_translation)
            # tbox.rotate(cs_rotation)
            # loc_pos = np.arctan2(tbox.center[1], tbox.center[0])
            # loc_yaw = quaternion_yaw(tbox.orientation)
            # wscore = 0.5 * (1.0 + np.cos(2.0 * (loc_yaw - loc_pos)))
            # dist = np.sqrt(tbox.center[0] ** 2 + tbox.center[1] ** 2)
            # trk.info.wscore = wscore
            # trk.info.dist = dist
            ########################################################

        # DATA ASSOCIATION
        if len(m_trks) > 0 and len(dets) > 0:
            m_trks = np.stack(m_trks, axis=0)
            m_trks_S = np.stack(m_trks_S, axis=0)
            a_trks = np.stack(a_trks, axis=0)
            a_trks_S = np.stack(a_trks_S, axis=0)
            m_dets = dets[:, self.tracker_generator.m_state_on, np.newaxis].copy()
            a_dets = dets[:, self.tracker_generator.a_state_on, np.newaxis].copy()

            matched, reverse, unmatched_dets, unmatched_trks = samac_associate(m_dets=m_dets, m_trks=m_trks, m_trks_S=m_trks_S,
                                                                               a_dets=a_dets, a_trks=a_trks, a_trks_S=a_trks_S,
                                                                               trackers=self.trackers,
                                                                               m_state_on=self.tracker_generator.m_state_on,
                                                                               a_state_on=self.tracker_generator.a_state_on,
                                                                               da_state_on=self.tracker_generator.da_state_on,
                                                                               m_yaw_pos=self.tracker_generator.m_yaw_pos, a_yaw_pos=self.tracker_generator.a_yaw_pos,
                                                                               mdist_thres=self.mdist_thres)
            # elif match_distance == 'mahal':
            #     comp_dets = dets[:, self.tracker_generator.state_on]
            #     comp_trks = trks[:, self.tracker_generator.state_on]
            #     comp_trks_S = np.zeros((len(trks), self.tracker_generator.dim_z, self.tracker_generator.dim_z))
            #     comp_trks_S[:, np.ones((self.tracker_generator.dim_z, self.tracker_generator.dim_z), dtype=bool)] = trks_S[:, self.tracker_generator.state_cov_on]
            #     matched, reverse, matched_dets, matched_trks, unmatched_dets, unmatched_trks = associate(dets=comp_dets, trks=comp_trks, trks_S=comp_trks_S, yaw_pos=self.tracker_generator.yaw_pos,
            #                                                              mahalanobis_threshold=self.mdist_thres,
            #                                                              print_debug=print_debug,
            #                                                              match_algorithm=match_algorithm)
            #     # relationship with matched and unmatched detection
            # elif match_distance == 'icp':
            #     matched, reverse, unmatched_dets, unmatched_trks = samac_associate(dets=dets, trks=trks, trks_S=trks_S, trackers=self.trackers, trks_info=trks_info, trks_tracking=trks_tracking, yaw_pos=self.tracker_generator.yaw_pos, mdist_thres=self.mdist_thres, iou_thres=0.1)


        else:
            matched, reverse, unmatched_dets, unmatched_trks = np.empty((0, 2), dtype=int), np.zeros((len(dets), len(m_trks)), dtype=np.bool), np.arange(len(dets)), np.arange(len(m_trks))


        for idx in matched:
            bbox3d = dets[idx[0], :]
            if reverse[idx[0], idx[1]]:
                bbox3d[yaw_pos] += np.pi
                bbox3d[yaw_pos] = angle_in_range(bbox3d[yaw_pos])
            self.trackers[idx[1]].info.update(info[idx[0], :])
            self.trackers[idx[1]].mtracker.update(bbox3d[self.tracker_generator.m_state_on])
            self.trackers[idx[1]].atracker.update(bbox3d[self.tracker_generator.a_state_on])


        # NEW TRACK GENERATION
        # matched detection overlap with unmatched detection or updated tracks? => delete unmatched detection # the certain ratio
        for i in unmatched_dets:
            trk = self.tracker_generator.generate_tracker(z=dets[i, :], info=info[i])
            self.trackers.append(trk)

        # TRACK MANAGEMENT
        i = len(self.trackers)
        # self.min_hits = 4
        for trk in reversed(self.trackers):
            i -= 1
            m_z = trk.mtracker.get_state()
            a_z = trk.atracker.get_state()
            bbox3d = np.concatenate((m_z, a_z))
            bbox3d = bbox3d[self.samac2dataset].reshape(1, -1).squeeze()

            # print(trk.info.info)
            if trk.info.tracking:
                if trk.info.time_since_update < self.max_track_age:
                    if trk.info.time_since_update == 0:
                        ret.append(np.concatenate((bbox3d, [trk.info.id + 1], trk.info.info)).reshape(1, -1))
                        prior.append(np.array([trk.info.m_prior, trk.info.a_prior]))
                else:
                    self.trackers.pop(i)
            else:
                # remove tracker right away when no two-times update # AP-wise correction
                if trk.info.time_since_update < self.max_tracklet_age:
                    if trk.info.hits >= self.min_hits or trk.info.hit_streak > 0:
                        trk.info.tracking = True
                    if trk.info.tracking or self.frame_count < self.min_hits:
                        ret.append(np.concatenate((bbox3d, [trk.info.id + 1], trk.info.info)).reshape(1, -1))
                        prior.append(np.array([trk.info.m_prior, trk.info.a_prior]))
                else:
                    self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret), np.stack(prior)  # x, y, z, theta, w, l, h, ID, other info, confidence
        return np.empty((0, 15 + 7)), np.empty((0, 2))