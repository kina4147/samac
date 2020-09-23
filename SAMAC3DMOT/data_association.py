import numpy as np

from .utils import angle_in_range #, corners, bbox_iou2d, bbox_iou3d, bbox_adjacency_2d, adjacency_2d

def greedy_match(distance_matrix, max_dist = 1000000):
    '''
    Find the one-to-one matching using greedy allgorithm choosing small distance
    distance_matrix: (num_detections, num_tracks)
    '''
    matched_indices = []

    num_detections, num_tracks = distance_matrix.shape
    distance_1d = distance_matrix.reshape(-1)
    index_1d = np.argsort(distance_1d)
    index_2d = np.stack([index_1d // num_tracks, index_1d % num_tracks], axis=1)
    detection_id_matches_to_tracking_id = [-1] * num_detections
    tracking_id_matches_to_detection_id = [-1] * num_tracks


    for sort_i in range(index_2d.shape[0]):
        detection_id = int(index_2d[sort_i][0])
        tracking_id = int(index_2d[sort_i][1])
        if tracking_id_matches_to_detection_id[tracking_id] == -1 and detection_id_matches_to_tracking_id[
            detection_id] == -1:
            tracking_id_matches_to_detection_id[tracking_id] = detection_id
            detection_id_matches_to_tracking_id[detection_id] = tracking_id
            if distance_matrix[detection_id, tracking_id] >= max_dist:
                break
            matched_indices.append([detection_id, tracking_id])

    matched_indices = np.array(matched_indices)
    return matched_indices

def estimate_likelihood(pred_z, z, S):
    np.exp(x-z)

def samac_associate(m_dets=None, m_trks=None, m_trks_S=None, a_dets=None, a_trks=None, a_trks_S=None, trackers=None, m_state_on=None, a_state_on=None, da_state_on=None, m_yaw_pos=2, a_yaw_pos=-1, mdist_thres=None):
    reverse_matrix = np.zeros((len(m_dets), len(m_trks)), dtype=np.bool)
    m_likelihood = np.full((len(m_dets), len(m_trks)), 1.0, dtype=np.float32)
    a_likelihood = np.full((len(m_dets), len(m_trks)), 1.0, dtype=np.float32)
    m_prior = np.zeros((len(m_trks)), dtype=np.float32)
    a_prior = np.zeros((len(m_trks)), dtype=np.float32)
    max_dist = 50.0
    min_dist = 0.0

    a_da_state_on = da_state_on[a_state_on].copy()
    a_dets = a_dets[:, a_da_state_on, :]
    a_trks = a_trks[:, a_da_state_on, :]
    dim_a_da = np.count_nonzero(a_da_state_on)
    a_da_trks_S = np.zeros((len(a_trks), dim_a_da, dim_a_da))
    a_da_trks_S[:, np.ones((dim_a_da, dim_a_da), dtype=bool)] = a_trks_S[:, a_da_state_on.reshape(-1, 1).dot(a_da_state_on.reshape(-1, 1).T)]
    for t, m_trk in enumerate(m_trks):
        # if trackers[t].info.wscore < 0.3:
        #     a_state_on[1] = False
        # elif trackers[t].info.wscore > 0.7:
        #     a_state_on[2] = False
        m_S_inv = np.linalg.inv(m_trks_S[t])
        a_S_inv = np.linalg.inv(a_da_trks_S[t])
        d_count = 0
        for d, m_det in enumerate(m_dets):
            m_diff = (m_det - m_trk).reshape((-1, 1))
            # manual reversed angle by 180 when diff > 90 or < -90 degree
            if -1 < m_yaw_pos < len(m_diff):
                m_diff[m_yaw_pos] = angle_in_range(m_diff[m_yaw_pos])
                if abs(m_diff[m_yaw_pos]) > np.pi / 2.0: # 180 deg diff
                    m_diff[m_yaw_pos] += np.pi
                    m_diff[m_yaw_pos] = angle_in_range(m_diff[m_yaw_pos])
                    reverse_matrix[d, t] = True
            m_mdist = np.sqrt(np.dot(np.dot(m_diff.T, m_S_inv), m_diff))
            if m_mdist < mdist_thres['motion'][len(m_diff)]:
                d_count += 1
                a_diff = (a_dets[d, :] - a_trks[t, :]).reshape((-1, 1))
                if -1 < a_yaw_pos < len(a_diff):
                    a_diff[a_yaw_pos] = angle_in_range(a_diff[a_yaw_pos])
                    if abs(a_diff[a_yaw_pos]) > np.pi / 2.0:  # 180 deg diff
                        a_diff[a_yaw_pos] += np.pi
                        a_diff[a_yaw_pos] = angle_in_range(a_diff[a_yaw_pos])
                        reverse_matrix[d, t] = True
                a_mdist = np.sqrt(np.dot(np.dot(a_diff.T, a_S_inv), a_diff))
                m_likelihood[d, t] = -(mdist_thres['motion'][len(m_diff)] - m_mdist)/mdist_thres['motion'][len(m_diff)]
                if a_mdist < mdist_thres['appearance'][len(a_diff)]:
                    a_likelihood[d, t] = -(mdist_thres['appearance'][len(a_diff)] - a_mdist)/mdist_thres['appearance'][len(a_diff)]
                else:
                    a_likelihood[d, t] = 0.0
                    # m_likelihood = -smn.pdf(det[0:3][m_state_on], trk[0:3][m_state_on], m_trk_S) # -np.exp(logpdf(det[0:3][m_state_on], trk[0:3][m_state_on], m_trk_S)) # -smn(trk[0:3][m_state_on], m_trk_S).pdf(det[0:3][m_state_on])
                # m_flike = fmn(det[0:3][m_state_on], trk[0:3][m_state_on], m_trk_S)
                # a_likelihood = -smn.pdf(det[3:7][a_state_on], trk[3:7][a_state_on], a_trk_S) # -np.exp(logpdf(det[3:7][a_state_on], trk[3:7][a_state_on], a_trk_S)) # -smn(trk[3:7][a_state_on], a_trk_S).pdf(det[3:7][a_state_on]))
                # a_flike = fmn(det[3:7][a_state_on], trk[3:7][a_state_on], a_trk_S)
        if d_count > 0:
            close_score = 1.0 # (max_dist - min((max_dist - min_dist), max(trackers[t].info.dist - min_dist, 0.0))) / max_dist
            crowd_score = 0.5 * (1.0 + min(3.0, d_count - 1)/3.0)
            rigid_score = 0.5 * (1.0 + len(a_diff)/5.0)
            a_prior[t] = 1.0 # 0.5 * close_score * crowd_score * rigid_score
            m_prior[t] = 1.0 # - a_prior[t]
            a_prior[t] = a_prior[t] * trackers[t].info.a_likelihood
            m_prior[t] = m_prior[t] * trackers[t].info.m_likelihood
            sum_prior = m_prior[t] + a_prior[t]
            a_prior[t] = a_prior[t] / sum_prior
            m_prior[t] = m_prior[t] / sum_prior
            trackers[t].info.m_prior = m_prior[t]
            trackers[t].info.a_prior = a_prior[t]
        likelihood = m_prior*m_likelihood + a_prior*a_likelihood
    match_algorithm = 'greedy'
    if match_algorithm == 'greedy':
        matched_indices = greedy_match(likelihood, 0.0)
    elif match_algorithm == 'hungarian':
        matched_indices = linear_assignment(likelihood)
        remove_indices = []
        for i, (midx, tidx) in enumerate(matched_indices):
            if likelihood[midx, tidx] > 0.0:
                remove_indices.append(i)
            matched_indices = np.delete(matched_indices, remove_indices, axis=0)
    else:
        pass

    unmatched_trks = np.arange(len(m_trks))
    unmatched_dets = np.arange(len(m_dets))
    if len(matched_indices) > 0:
        matched_dets = matched_indices[:, 0]
        matched_trks = matched_indices[:, 1]
        unmatched_dets = np.delete(unmatched_dets, matched_dets)
        unmatched_trks = np.delete(unmatched_trks, matched_trks)
        # likelihood estimation for next data association
        for midx in matched_indices:
            trackers[midx[1]].info.m_likelihood = -m_likelihood[midx[0], midx[1]]# / (m_likelihood[midx[0], midx[1]] + a_likelihood[midx[0], midx[1]])
            trackers[midx[1]].info.a_likelihood = -a_likelihood[midx[0], midx[1]]# / (m_likelihood[midx[0], midx[1]] + a_likelihood[midx[0], midx[1]])



    return matched_indices, reverse_matrix, unmatched_dets, unmatched_trks