import numpy as np

NUSCENES_TRACKING_NAMES = [
    'bicycle',
    'bus',
    'car',
    'motorcycle',
    'pedestrian',
    'trailer',
    'truck'
]

# self.AP = {
#     'bicycle': 0.223,
#     'bus': 0.549,
#     'car': 0.811,
#     'motorcycle': 0.515,
#     'pedestrian': 0.801,
#     'trailer': 0.429,
#     'truck': 0.485
#
# }
class MahalanobisDistThres(object):

    def __init__(self):
        # self.mdist_threshold ={'motion': [], 'appearance': []}
        self.mdist_threshold = {'motion': {}, 'appearance': {}}
        # 99%
        # self.mdist_threshold['motion'][1] = 6.635
        # self.mdist_threshold['motion'][2] = 9.21
        # self.mdist_threshold['motion'][3] = 11.345
        # self.mdist_threshold['motion'][4] = 13.277
        # self.mdist_threshold['motion'][5] = 15.086
        # self.mdist_threshold['motion'][6] = 16.812
        # self.mdist_threshold['motion'][7] = 18.475
        # 90%
        # self.mdist_threshold['motion'][1] = 2.706
        # self.mdist_threshold['motion'][2] = 4.605 # 4.1 #
        # self.mdist_threshold['motion'][3] = 6.251 # 5.6 #
        # self.mdist_threshold['motion'][4] = 7.779 # 7.0 #7.14 #
        # self.mdist_threshold['motion'][5] = 9.2361 # 8.4 #
        # self.mdist_threshold['motion'][6] = 10.645 # 9.7 #
        # self.mdist_threshold['motion'][7] = 12.017 #11.0 # 12.017# 11.0 #
        # 95%
        self.mdist_threshold['motion'][1] = 3.841
        self.mdist_threshold['motion'][2] = 5.991
        self.mdist_threshold['motion'][3] = 7.815
        self.mdist_threshold['motion'][4] = 9.448
        self.mdist_threshold['motion'][5] = 11.071
        self.mdist_threshold['motion'][6] = 12.592
        self.mdist_threshold['motion'][7] = 14.067
        # 90%
        # self.mdist_threshold['appearance'][1] = 2.706
        # self.mdist_threshold['appearance'][2] = 4.605 # 4.1 #
        # self.mdist_threshold['appearance'][3] = 6.251 # 5.6 #
        # self.mdist_threshold['appearance'][4] = 7.779 # 7.0 #7.14 #
        # self.mdist_threshold['appearance'][5] = 9.2361 # 8.4 #
        # self.mdist_threshold['appearance'][6] = 10.645 # 9.7 #
        # self.mdist_threshold['appearance'][7] = 12.017 #11.0 # 12.017# 11.0 #
        # 95%
        self.mdist_threshold['appearance'][1] = 3.841
        self.mdist_threshold['appearance'][2] = 5.991
        self.mdist_threshold['appearance'][3] = 7.815
        self.mdist_threshold['appearance'][4] = 9.448
        self.mdist_threshold['appearance'][5] = 11.071
        self.mdist_threshold['appearance'][6] = 12.592
        self.mdist_threshold['appearance'][7] = 14.067

class Covariance(object):
  '''
  Define different Kalman Filter covariance matrix
  Kalman Filter states:
  [x, y, z, rot_y, l, w, h, x_dot, y_dot, z_dot]
  '''
  def __init__(self, covariance_id, tracking_name):
    if covariance_id == -1:
        m_head_P = {'Car': [0.01573432464782778, 0.02262684396208941, 0.20905808193001274, 0.014430059309977586,
                            0.2725462091572646],
                    'Pedestrian': [0.0027934371681371663, 0.0025811489821510286, 0.3661471166989713,
                                   0.00179967550414337, 0.4407226652352146],
                    'Cyclist': [0.002898954008057376, 0.0019365279339227214, 0.008908677425920723,
                                0.0034119858208359367, 0.009060468411393583]}
        m_head_Q = {'Car': [0.10380455999713374, 0.42863869975101077, 0.016831905906731386, 0.006782467771905974,
                            0.03424172052510986],
                    'Pedestrian': [0.009111745957192961, 0.005405285008465497, 0.19054965115680692,
                                   0.001186317626907526, 0.39871098857720394],
                    'Cyclist': [0.08224959278709605, 0.016694239840891908, 0.08557781173590637, 0.0010526939674074627,
                                0.1752942091402293]}
        m_no_head_P = {'Car': [0.01573432464782778, 0.02262684396208941, 0.20905808193001274, 0.010821999823546256,
                               0.011569574525854488, 0.2725462091572646],
                       'Pedestrian': [0.0027934371681371663, 0.0025811489821510286, 0.3661471166989713,
                                      0.0018958826568090392, 0.0018816147349337419, 0.4407226652352146],
                       'Cyclist': [0.002898954008057376, 0.0019365279339227214, 0.008908677425920723,
                                   0.002903121163585994, 0.0018540004387348063, 0.009060468411393583]}
        m_no_head_Q = {'Car': [0.10380455999713374, 0.42863869975101077, 0.016831905906731386, 0.00463987755222138,
                               0.006366154414340404, 0.03424172052510986],
                       'Pedestrian': [0.009111745957192961, 0.005405285008465497, 0.19054965115680692,
                                      0.0009150394251635038, 0.0013419162407636962, 0.39871098857720394],
                       'Cyclist': [0.08224959278709605, 0.016694239840891908, 0.08557781173590637,
                                   0.0007829263466402028, 0.0012537651242253874, 0.1752942091402293]}
        m_R = {'Car': [0.01573432464782778, 0.02262684396208941, 0.20905808193001274],
               'Pedestrian': [0.0027934371681371663, 0.0025811489821510286, 0.3661471166989713],
               'Cyclist': [0.002898954008057376, 0.0019365279339227214, 0.008908677425920723]}
        a_P = {'Car': [0.20905808193001274, 0.008085391297876282, 0.010006356250151227, 0.08507043608213549,
                       0.009748388056278999, 0.00339334962229757],
               'Pedestrian': [0.3661471166989713, 0.0034891803982593688, 0.015357421043752787, 0.0208315437306977,
                              0.007513134956503478, 0.0024644369752963146],
               'Cyclist': [0.008908677425920723, 0.003979309069356323, 0.010237486427632865, 0.008525975134736985,
                           0.006965698543746481, 0.00245144791737919]}
        a_Q = {'Car': [0.016831905906731386, 0.0010242622599452144, 0.0, 0.0, 0.0, 0.0018291409634644987],
               'Pedestrian': [0.19054965115680692, 0.00040973104639999516, 0.0, 0.0, 0.0, 0.0004770570685061522],
               'Cyclist': [0.08557781173590637, 0.000426128740722422, 0.0, 0.0, 0.0, 0.0006992271505930039]}
        a_R = {'Car': [0.20905808193001274, 0.008085391297876282, 0.010006356250151227, 0.08507043608213549,
                       0.009748388056278999],
               'Pedestrian': [0.3661471166989713, 0.0034891803982593688, 0.015357421043752787, 0.0208315437306977,
                              0.007513134956503478],
               'Cyclist': [0.008908677425920723, 0.003979309069356323, 0.010237486427632865, 0.008525975134736985,
                           0.006965698543746481]}

    elif covariance_id == 0:
        m_head_P = {
        'Car': [0.05, 0.05, 0.5, 0.05, 0.5], # [0.01969623, 0.01179107, 0.52534431, 0.01334779, 0.52534431], # [0.08900372, 0.09412005, 1.00535696, 0.115581232, 0.99492726],
        'Cyclist': [0.02, 0.02, 0.6, 0.02, 0.6], # [0.04052819, 0.0398904, 1.06442726, 0.061505764, 1.30414345],
        'Pedestrian': [0.01, 0.02, 0.7, 0.01, 0.7] # 0.03855275, 0.0377111, 2.0751833, 0.058906636, 2.0059979]
        }

        m_head_Q = {
        'Car': [0.005, 0.005, 0.1, 0.005, 0.1], #[2.94827444e-03, 2.18784125e-03, 1.10964054e-01, 2.94827444e-03, 1.10964054e-01], #[1.58918523e-01, 1.24935318e-01, 9.22800791e-02, 0.202148289, 9.22800791e-02],
        'Cyclist': [0.002, 0.002, 0.2, 0.002, 0.2], #[3.23647590e-02, 3.86650974e-02, 2.34967407e-01, 0.050422885, 2.34967407e-01],
        'Pedestrian': [0.001, 0.001, 0.3, 0.001, 0.3] #[3.34814566e-02, 2.47354921e-02, 4.24962535e-01, 0.041627545, 4.24962535e-01]
        }

        m_R = {
        'Car': [0.05, 0.05, 0.5], #[0.01969623, 0.01179107, 0.52534431], # [0.08900372, 0.09412005, 1.00535696],
        'Cyclist': [0.02, 0.02, 0.6], # [0.04052819, 0.0398904, 1.06442726],
        'Pedestrian': [0.01, 0.01, 0.7] # [0.03855275, 0.0377111, 2.0751833]
        }

        #Kalman Filter state: [x, y, yaw, x', y', yaw']
        m_no_head_P = {
        'Car': [0.05, 0.05, 0.5, 0.05, 0.05, 0.5], # [0.08900372, 0.09412005, 1.00535696, 0.08120681, 0.08224643, 0.99492726],
        'Cyclist': [0.02, 0.02, 0.5, 0.02, 0.02, 0.5], # [0.04052819, 0.0398904, 1.06442726, 0.0437039, 0.04327734, 1.30414345],
        'Pedestrian': [0.01, 0.01, 0.5, 0.01, 0.01, 0.5] # [0.03855275, 0.0377111, 2.0751833, 0.04237008, 0.04092393, 2.0059979]
        }

        m_no_head_Q = {
        'Car': [0.003, 0.003, 0.1, 0.003, 0.003, 0.1], #[1.58918523e-01, 1.24935318e-01, 9.22800791e-02, 1.58918523e-01, 1.24935318e-01, 9.22800791e-02],
        'Cyclist': [0.002, 0.002, 0.2, 0.002, 0.002, 0.2], #[3.23647590e-02, 3.86650974e-02, 2.34967407e-01, 3.23647590e-02, 3.86650974e-02, 2.34967407e-01],
        'Pedestrian': [0.001, 0.001, 0.3, 0.001, 0.001, 0.3] #[3.34814566e-02, 2.47354921e-02, 4.24962535e-01, 3.34814566e-02, 2.47354921e-02, 4.24962535e-01]
        }

        a_P = {
        'Car': [0.52534431, 0.04189842, 0.00983173, 0.11816206, 0.01602004, 0.01837525], # [1.00535696, 0.03265469, 0.02359175, 0.10912802, 0.02455134, 0.02266425],
        'Cyclist': [0.6, 0.03, 0.01, 0.1, 0.01, 0.01], # [1.06442726, 0.01511711, 0.00957574, 0.03291016, 0.0111605, 0.01465631],
        'Pedestrian': [0.6, 0.03, 0.01, 0.1, 0.01, 0.01], # [2.0751833, 0.02482115, 0.0136347, 0.02286483, 0.0203149, 0.01482923]
        }

        a_Q = {
        'Car': [1.10964054e-01, 6.85044585e-03, 0, 0, 0, 6.85044585e-03], #[9.22800791e-02, 5.35573165e-03, 0, 0, 0, 5.35573165e-03],
        'Cyclist': [0.2, 0.005, 0, 0, 0, 0.005], #[2.34967407e-01, 5.47421635e-03, 0, 0, 0, 5.47421635e-03],
        'Pedestrian': [0.3, 0.004, 0, 0, 0, 0.004], #[4.24962535e-01, 5.94592529e-03, 0, 0, 0, 5.94592529e-03]
        }

        a_R = {
        'Car': [0.52534431, 0.04189842, 0.00983173, 0.11816206, 0.01602004], # [1.00535696, 0.03265469, 0.02359175, 0.10912802, 0.02455134],
        'Cyclist': [0.6, 0.03, 0.01, 0.1, 0.01], # [1.06442726, 0.01511711, 0.00957574, 0.03291016, 0.0111605],
        'Pedestrian': [0.6, 0.03, 0.01, 0.1, 0.01], # [2.0751833, 0.02482115, 0.0136347, 0.02286483, 0.0203149]
        }
    elif covariance_id == 1:
        # nuscenes # x, y, theta, v, theta'
        m_head_P = {
        'bicycle': [0.05390982, 0.05039431, 1.29464435, 0.06130649, 1.21635902],
        'bus': [0.17546469, 0.13818929, 0.1979503, 0.175599858, 0.22529652],
        'car': [0.08900372, 0.09412005, 1.00535696, 0.115581232, 0.99492726],
        'motorcycle': [0.04052819, 0.0398904, 1.06442726, 0.061505764, 1.30414345],
        'pedestrian': [0.03855275, 0.0377111, 2.0751833, 0.058906636, 2.0059979],
        'trailer': [0.23228021, 0.22229261, 1.05163481, 0.290263582, 0.97082174],
        'truck': [0.14862173, 0.1444596, 0.73122169, 0.148047001, 0.76188901]
        }

        m_head_Q = {
        'bicycle': [1.98881347e-02, 1.36552276e-02, 1.33430252e-01, 0.024124741, 1.33430252e-01],
        'bus': [1.17729925e-01, 8.84659079e-02, 2.09050032e-01, 0.147263546, 2.09050032e-01],
        'car': [1.58918523e-01, 1.24935318e-01, 9.22800791e-02, 0.202148289, 9.22800791e-02],
        'motorcycle': [3.23647590e-02, 3.86650974e-02, 2.34967407e-01, 0.050422885, 2.34967407e-01],
        'pedestrian': [3.34814566e-02, 2.47354921e-02, 4.24962535e-01, 0.041627545, 4.24962535e-01],
        'trailer': [4.19985099e-02, 3.68661552e-02, 5.63166240e-02, 0.055883703, 5.63166240e-02],
        'truck': [9.45275998e-02, 9.45620374e-02, 1.41680460e-01, 0.133706567, 1.41680460e-01]
        }

        m_R = {
        'bicycle': [0.05390982, 0.05039431, 1.29464435],
        'bus': [0.17546469, 0.13818929, 0.1979503],
        'car': [0.08900372, 0.09412005, 1.00535696],
        'motorcycle': [0.04052819, 0.0398904, 1.06442726],
        'pedestrian': [0.03855275, 0.0377111, 2.0751833],
        'trailer': [0.23228021, 0.22229261, 1.05163481],
        'truck': [0.14862173, 0.1444596, 0.73122169]
        }

        #Kalman Filter state: [x, y, yaw, x', y', yaw']
        m_no_head_P = {
        'bicycle': [0.05390982, 0.05039431, 1.29464435, 0.04560422, 0.04097244, 1.21635902],
        'bus': [0.17546469, 0.13818929, 0.1979503, 0.13263319, 0.11508148, 0.22529652],
        'car': [0.08900372, 0.09412005, 1.00535696, 0.08120681, 0.08224643, 0.99492726],
        'motorcycle': [0.04052819, 0.0398904, 1.06442726, 0.0437039, 0.04327734, 1.30414345],
        'pedestrian': [0.03855275, 0.0377111, 2.0751833, 0.04237008, 0.04092393, 2.0059979],
        'trailer': [0.23228021, 0.22229261, 1.05163481, 0.2138643, 0.19625241, 0.97082174],
        'truck': [0.14862173, 0.1444596, 0.73122169, 0.10683797, 0.10248689, 0.76188901]
        }

        m_no_head_Q = {
        'bicycle': [1.98881347e-02, 1.36552276e-02, 1.33430252e-01, 1.98881347e-02, 1.36552276e-02, 1.33430252e-01],
        'bus': [1.17729925e-01, 8.84659079e-02, 2.09050032e-01, 1.17729925e-01, 8.84659079e-02, 2.09050032e-01],
        'car': [1.58918523e-01, 1.24935318e-01, 9.22800791e-02, 1.58918523e-01, 1.24935318e-01, 9.22800791e-02],
        'motorcycle': [3.23647590e-02, 3.86650974e-02, 2.34967407e-01, 3.23647590e-02, 3.86650974e-02, 2.34967407e-01],
        'pedestrian': [3.34814566e-02, 2.47354921e-02, 4.24962535e-01, 3.34814566e-02, 2.47354921e-02, 4.24962535e-01],
        'trailer': [4.19985099e-02, 3.68661552e-02, 5.63166240e-02, 4.19985099e-02, 3.68661552e-02, 5.63166240e-02],
        'truck': [9.45275998e-02, 9.45620374e-02, 1.41680460e-01, 9.45275998e-02, 9.45620374e-02, 1.41680460e-01]
        }

        a_P = {
        'bicycle': [1.29464435, 0.01863044, 0.01169572, 0.02713823, 0.01295084, 0.01725477],
        'bus': [0.1979503, 0.05947248, 0.05507407, 0.78867322, 0.06684149, 0.05033665],
        'car': [1.00535696, 0.03265469, 0.02359175, 0.10912802, 0.02455134, 0.02266425],
        'motorcycle': [1.06442726, 0.01511711, 0.00957574, 0.03291016, 0.0111605, 0.01465631],
        'pedestrian': [2.0751833, 0.02482115, 0.0136347, 0.02286483, 0.0203149, 0.01482923],
        'trailer': [1.05163481, 0.07006275, 0.06354783, 1.37451601, 0.10500918, 0.05231335],
        'truck': [0.14862173, 0.1444596, 0.73122169, 0.05417157, 0.05484365, 0.69387238, 0.07748085, 0.0378078]
        }

        a_Q = {
        'bicycle': [1.33430252e-01, 5.10175742e-03, 0, 0, 0, 5.10175742e-03],
        'bus': [2.09050032e-01, 1.17616440e-02, 0, 0, 0, 1.17616440e-02],
        'car': [9.22800791e-02, 5.35573165e-03, 0, 0, 0, 5.35573165e-03],
        'motorcycle': [2.34967407e-01, 5.47421635e-03, 0, 0, 0, 5.47421635e-03],
        'pedestrian': [4.24962535e-01, 5.94592529e-03, 0, 0, 0, 5.94592529e-03],
        'trailer': [5.63166240e-02, 1.19415050e-02, 0, 0, 0, 1.19415050e-02],
        'truck': [1.41680460e-01, 8.38061721e-03, 0, 0, 0, 8.38061721e-03]
        }

        a_R = {
        'bicycle': [1.29464435, 0.01863044, 0.01169572, 0.02713823, 0.01295084],
        'bus': [0.1979503, 0.05947248, 0.05507407, 0.78867322, 0.06684149],
        'car': [1.00535696, 0.03265469, 0.02359175, 0.10912802, 0.02455134],
        'motorcycle': [1.06442726, 0.01511711, 0.00957574, 0.03291016, 0.0111605],
        'pedestrian': [2.0751833, 0.02482115, 0.0136347, 0.02286483, 0.0203149],
        'trailer': [1.05163481, 0.07006275, 0.06354783, 1.37451601, 0.10500918],
        'truck': [0.14862173, 0.1444596, 0.73122169, 0.05417157, 0.05484365, 0.69387238, 0.07748085]
        }
    elif covariance_id == 2: # train
        m_head_P = {'bicycle': [0.03140886233769575, 0.030032329603297745, 0.6651864446593515, 0.036822007935248105,
                                0.7178619380874288],
                    'bus': [0.17652309922789083, 0.13989355221437247, 0.05645355114189658, 0.1874482121791733,
                            0.06058940437154568],
                    'car': [0.09108666765010379, 0.09763363127076839, 0.7007670273595883, 0.12226477553638768,
                            0.6996158199707178],
                    'motorcycle': [0.033675482259082234, 0.031015023376216144, 0.27759691093927286, 0.04832244204138485,
                                   0.35028761504417105],
                    'pedestrian': [0.01506565209580167, 0.014796385473011108, 0.7287866228515731, 0.014584918452819338,
                                   0.7283704218524243],
                    'trailer': [0.24429138397649092, 0.22520932525696172, 0.8911764612947567, 0.2466585805014609,
                                0.8075854296013687],
                    'truck': [0.15309792841538286, 0.14984980710684842, 0.38189115443253085, 0.1567680847040434,
                              0.4083028892394904]}
        m_head_Q = {'bicycle': [0.017575825483969573, 0.012514663441570028, 0.008122685587104795, 0.02023260074945087,
                                0.008122685587104795],
                    'bus': [0.10607459350385301, 0.08525198236999426, 0.00027537687739734946, 0.13672041450078096,
                            0.00027537687739734946],
                    'car': [0.1407541534908422, 0.11236393462201978, 0.0005147269684307482, 0.1562874577152961,
                            0.0005147269684307482],
                    'motorcycle': [0.0324330607371846, 0.03883446723749115, 0.0010462697413800891, 0.04711464593020927,
                                   0.0010462697413800891],
                    'pedestrian': [0.033215941639602384, 0.02460987708542625, 0.009808133067737866, 0.0363253341003307,
                                   0.009808133067737866],
                    'trailer': [0.038109321011015364, 0.032485872221263534, 0.0018563427327925584, 0.046565708117849276,
                                0.0018563427327925584],
                    'truck': [0.08556833350518066, 0.0864937748092013, 0.00026775408936091253, 0.09065700335613833,
                              0.00026775408936091253]}
        m_no_head_P = {'bicycle': [0.03140886233769575, 0.030032329603297745, 0.6651864446593515, 0.02775654695466487,
                                   0.025859343952554518, 0.7178619380874288],
                       'bus': [0.17652309922789083, 0.13989355221437247, 0.05645355114189658, 0.13435156875889667,
                               0.11553421059889743, 0.06058940437154568],
                       'car': [0.09108666765010379, 0.09763363127076839, 0.7007670273595883, 0.08328681651884613,
                               0.08648366331460586, 0.6996158199707178],
                       'motorcycle': [0.033675482259082234, 0.031015023376216144, 0.27759691093927286,
                                      0.03512990504089385, 0.032538557303022526, 0.35028761504417105],
                       'pedestrian': [0.01506565209580167, 0.014796385473011108, 0.7287866228515731,
                                      0.013351432865254226, 0.013167331055154677, 0.7283704218524243],
                       'trailer': [0.24429138397649092, 0.22520932525696172, 0.8911764612947567, 0.2291199591182645,
                                   0.20108928061011702, 0.8075854296013687],
                       'truck': [0.15309792841538286, 0.14984980710684842, 0.38189115443253085, 0.11227869011414511,
                                 0.10936236821963524, 0.4083028892394904]}
        m_no_head_Q = {
            'bicycle': [0.017575825483969573, 0.012514663441570028, 0.008122685587104795, 0.017575825483969573,
                        0.012514663441570028, 0.008122685587104795],
            'bus': [0.10607459350385301, 0.08525198236999426, 0.00027537687739734946, 0.10607459350385301,
                    0.08525198236999426, 0.00027537687739734946],
            'car': [0.1407541534908422, 0.11236393462201978, 0.0005147269684307482, 0.1407541534908422,
                    0.11236393462201978, 0.0005147269684307482],
            'motorcycle': [0.0324330607371846, 0.03883446723749115, 0.0010462697413800891, 0.0324330607371846,
                           0.03883446723749115, 0.0010462697413800891],
            'pedestrian': [0.033215941639602384, 0.02460987708542625, 0.009808133067737866, 0.033215941639602384,
                           0.02460987708542625, 0.009808133067737866],
            'trailer': [0.038109321011015364, 0.032485872221263534, 0.0018563427327925584, 0.038109321011015364,
                        0.032485872221263534, 0.0018563427327925584],
            'truck': [0.08556833350518066, 0.0864937748092013, 0.00026775408936091253, 0.08556833350518066,
                      0.0864937748092013, 0.00026775408936091253]}
        m_R = {'bicycle': [0.03140886233769575, 0.030032329603297745, 0.6651864446593515],
               'bus': [0.17652309922789083, 0.13989355221437247, 0.05645355114189658],
               'car': [0.09108666765010379, 0.09763363127076839, 0.7007670273595883],
               'motorcycle': [0.033675482259082234, 0.031015023376216144, 0.27759691093927286],
               'pedestrian': [0.01506565209580167, 0.014796385473011108, 0.7287866228515731],
               'trailer': [0.24429138397649092, 0.22520932525696172, 0.8911764612947567],
               'truck': [0.15309792841538286, 0.14984980710684842, 0.38189115443253085]}
        a_P = {'bicycle': [0.017651872243258978, 0.011129839173051045, 0.026859657108696713, 0.012189447548468613,
                           0.016591538840345976],
               'bus': [0.06000910432223465, 0.05498421987675385, 0.7891502541938713, 0.06691417913951915,
                       0.051341747933287434],
               'car': [0.034786411857449843, 0.023690968627490695, 0.1108406560994001, 0.024957533851838486,
                       0.024436994468799253],
               'motorcycle': [0.014862404975815303, 0.00946003644908104, 0.03284368129818355, 0.010923536627530386,
                              0.014169494496030253],
               'pedestrian': [0.02366530890354684, 0.013571724992689575, 0.022822947414839426, 0.019634876969502765,
                              0.013879729104407448],
               'trailer': [0.0734285203491551, 0.06393051257011621, 1.427678977313346, 0.10480254618777335,
                           0.05513978805809601],
               'truck': [0.05645596551974882, 0.05539375897235971, 0.7157749901276627, 0.08015750319121716,
                         0.039607142634289126]}
        a_Q = {'bicycle': [0.004614341147122671, 0.0, 0.0, 0.0, 0.004614341147122671],
               'bus': [0.010634614494109707, 0.0, 0.0, 0.0, 0.010634614494109707],
               'car': [0.00483352613623634, 0.0, 0.0, 0.0, 0.00483352613623634],
               'motorcycle': [0.004952924557143415, 0.0, 0.0, 0.0, 0.004952924557143415],
               'pedestrian': [0.00556782701029592, 0.0, 0.0, 0.0, 0.00556782701029592],
               'trailer': [0.009796590137550265, 0.0, 0.0, 0.0, 0.009796590137550265],
               'truck': [0.00772167266054647, 0.0, 0.0, 0.0, 0.00772167266054647]}
        a_R = {'bicycle': [0.017651872243258978, 0.011129839173051045, 0.026859657108696713, 0.012189447548468613],
               'bus': [0.06000910432223465, 0.05498421987675385, 0.7891502541938713, 0.06691417913951915],
               'car': [0.034786411857449843, 0.023690968627490695, 0.1108406560994001, 0.024957533851838486],
               'motorcycle': [0.014862404975815303, 0.00946003644908104, 0.03284368129818355, 0.010923536627530386],
               'pedestrian': [0.02366530890354684, 0.013571724992689575, 0.022822947414839426, 0.019634876969502765],
               'trailer': [0.0734285203491551, 0.06393051257011621, 1.427678977313346, 0.10480254618777335],
               'truck': [0.05645596551974882, 0.05539375897235971, 0.7157749901276627, 0.08015750319121716]}

    elif covariance_id == 3: # val
        m_head_P = {
            'bicycle': [0.0334228511795664, 0.035880661951437194, 1.2450085067819545, 0.04048809436140954, 1.4100296512550243],
            'bus': [0.2866325673876865, 0.20302860648305804, 0.2899626942265115, 0.252681098964695, 0.2650967889118452],
            'car': [0.10955980088983346, 0.11056225020923503, 0.8612540352046152, 0.1459412687955439, 0.7980710480228093],
            'motorcycle': [0.07936535832144131, 0.07043775813175202, 0.9105783921203751, 0.10890598539860287, 0.8176823034861685],
            'pedestrian': [0.030151095523957394, 0.030464043698484034, 0.9653376119006691, 0.0400366988586378,0.8370134899267443],
            'trailer': [0.45890664295266254, 0.3926525898122129, 2.563010733423147, 0.29240554836016286, 1.1818636006833805],
            'truck': [0.21026759028390912, 0.18005625717524637, 0.5990509230914925, 0.1737233550741575, 0.610185105315484]}
        m_head_Q = {
            'bicycle': [0.021813495711778753, 0.04813367226255504, 0.000849021102071693, 0.029211976268996666, 0.000849021102071693],
            'bus': [0.2047641548427021, 0.08010485552272169, 0.00041099616007820374, 0.1692410720625874,0.00041099616007820374],
            'car': [0.061902233491183054, 0.04918006266657506, 0.0005057323558213683, 0.08263120214628222,0.0005057323558213683],
            'motorcycle': [0.03944293990819283, 0.03558182702052929, 0.002003853443360149, 0.0511514291389453, 0.002003853443360149],
            'pedestrian': [0.02040701438905246, 0.03094954307028024, 0.009777551635026796, 0.03323052807366408, 0.009777551635026796],
            'trailer': [0.0407261584687881, 0.03575475492919131, 0.00029980416189495624, 0.04339965490589297, 0.00029980416189495624],
            'truck': [0.05316553102823158, 0.04368352512517618, 0.0004263686604175415, 0.07754638438443205, 0.0004263686604175415]}
        m_no_head_P = {
            'bicycle': [0.0334228511795664, 0.035880661951437194, 1.2450085067819545, 0.027530684262965663, 0.03113310974651576, 1.4100296512550243],
           'bus': [0.2866325673876865, 0.20302860648305804, 0.2899626942265115, 0.1786085606843445, 0.14958948964600935, 0.2650967889118452],
           'car': [0.10955980088983346, 0.11056225020923503, 0.8612540352046152, 0.09615893873459985, 0.1003307252025598, 0.7980710480228093],
           'motorcycle': [0.07936535832144131, 0.07043775813175202, 0.9105783921203751, 0.07318863186550931, 0.06963691637659275, 0.8176823034861685],
           'pedestrian': [0.030151095523957394, 0.030464043698484034, 0.9653376119006691, 0.028675502235130868, 0.03174146651629779, 0.8370134899267443],
           'trailer': [0.45890664295266254, 0.3926525898122129, 2.563010733423147, 0.30998879239793514, 0.26435348533791275, 1.1818636006833805],
           'truck': [0.21026759028390912, 0.18005625717524637, 0.5990509230914925, 0.14112129576158794, 0.1110399199868701, 0.610185105315484]}
        m_no_head_Q = {
            'bicycle': [0.021813495711778753, 0.04813367226255504, 0.000849021102071693, 0.021813495711778753, 0.04813367226255504, 0.000849021102071693],
            'bus': [0.2047641548427021, 0.08010485552272169, 0.00041099616007820374, 0.2047641548427021, 0.08010485552272169, 0.00041099616007820374],
            'car': [0.061902233491183054, 0.04918006266657506, 0.0005057323558213683, 0.061902233491183054, 0.04918006266657506, 0.0005057323558213683],
            'motorcycle': [0.03944293990819283, 0.03558182702052929, 0.002003853443360149, 0.03944293990819283, 0.03558182702052929, 0.002003853443360149],
            'pedestrian': [0.02040701438905246, 0.03094954307028024, 0.009777551635026796, 0.02040701438905246, 0.03094954307028024, 0.009777551635026796],
            'trailer': [0.0407261584687881, 0.03575475492919131, 0.00029980416189495624, 0.0407261584687881, 0.03575475492919131, 0.00029980416189495624],
            'truck': [0.05316553102823158, 0.04368352512517618, 0.0004263686604175415, 0.05316553102823158, 0.04368352512517618, 0.0004263686604175415]}
        m_R = {'bicycle': [0.0334228511795664, 0.035880661951437194, 1.2450085067819545],
               'bus': [0.2866325673876865, 0.20302860648305804, 0.2899626942265115],
               'car': [0.10955980088983346, 0.11056225020923503, 0.8612540352046152],
               'motorcycle': [0.07936535832144131, 0.07043775813175202, 0.9105783921203751],
               'pedestrian': [0.030151095523957394, 0.030464043698484034, 0.9653376119006691],
               'trailer': [0.45890664295266254, 0.3926525898122129, 2.563010733423147],
               'truck': [0.21026759028390912, 0.18005625717524637, 0.5990509230914925]}
        a_P = {'bicycle': [1.2450085067819545, 0.023878749575495743, 0.01766769282134109, 0.04775547460430642, 0.020566097219027604, 0.016830876422964287],
               'bus': [0.2899626942265115, 0.09419092344847287, 0.07122776988429849, 1.6132907767390896, 0.17916283188880605,
                       0.056945976352654006],
               'car': [0.8612540352046152, 0.03831866634449853, 0.02543066025998946, 0.11876370206490382, 0.026331255430681468,
                       0.02460494733761304],
               'motorcycle': [0.9105783921203751, 0.04017070885282749, 0.02191565883523029, 0.0661071631246177, 0.020726136837330225,
                              0.01762362141326559],
               'pedestrian': [0.9653376119006691, 0.02348479133607479, 0.014482736798577806, 0.029080146870617864, 0.019939854134655682,
                              0.013791007614644954],
               'trailer': [2.563010733423147, 0.13098796930671053, 0.11966790114943467, 3.4663987710623663, 0.21818734057883593,
                           0.07119821533574046],
               'truck': [0.5990509230914925, 0.06738495366560077, 0.06663293137298663, 1.1663729276260995, 0.11774305611618718,
                         0.03892779228916867]}
        a_Q = {'bicycle': [0.000849021102071693, 0.0044996926254524125, 0.0, 0.0, 0.0, 0.0044996926254524125],
               'bus': [0.00041099616007820374, 0.009349719080974699, 0.0, 0.0, 0.0, 0.009349719080974699],
               'car': [0.0005057323558213683, 0.006290630244676732, 0.0, 0.0, 0.0, 0.006290630244676732],
               'motorcycle': [0.002003853443360149, 0.006049322153801102, 0.0, 0.0, 0.0, 0.006049322153801102],
               'pedestrian': [0.009777551635026796, 0.005870610955345415, 0.0, 0.0, 0.0, 0.005870610955345415],
               'trailer': [0.00029980416189495624, 0.010273108457136741, 0.0, 0.0, 0.0, 0.010273108457136741],
               'truck': [0.0004263686604175415, 0.008091075786808684, 0.0, 0.0, 0.0, 0.008091075786808684]}
        a_R = {'bicycle': [1.2450085067819545, 0.023878749575495743, 0.01766769282134109, 0.04775547460430642, 0.020566097219027604],
               'bus': [0.2899626942265115, 0.09419092344847287, 0.07122776988429849, 1.6132907767390896, 0.17916283188880605],
               'car': [0.8612540352046152, 0.03831866634449853, 0.02543066025998946, 0.11876370206490382, 0.026331255430681468],
               'motorcycle': [0.9105783921203751, 0.04017070885282749, 0.02191565883523029, 0.0661071631246177, 0.020726136837330225],
               'pedestrian': [0.9653376119006691, 0.02348479133607479, 0.014482736798577806, 0.029080146870617864, 0.019939854134655682],
               'trailer': [2.563010733423147, 0.13098796930671053, 0.11966790114943467, 3.4663987710623663, 0.21818734057883593],
               'truck': [0.5990509230914925, 0.06738495366560077, 0.06663293137298663, 1.1663729276260995, 0.11774305611618718]}

    else:
      assert(False)

    # self.m_head_P = {tracking_name: np.diag(m_head_P[tracking_name]) for tracking_name in NUSCENES_TRACKING_NAMES}
    # self.m_head_Q = {tracking_name: np.diag(m_head_Q[tracking_name]) for tracking_name in NUSCENES_TRACKING_NAMES}
    # self.m_no_head_P = {tracking_name: np.diag(m_no_head_P[tracking_name]) for tracking_name in NUSCENES_TRACKING_NAMES}
    # self.m_no_head_Q = {tracking_name: np.diag(m_no_head_Q[tracking_name]) for tracking_name in NUSCENES_TRACKING_NAMES}
    # self.m_R = {tracking_name: np.diag(m_R[tracking_name]) for tracking_name in NUSCENES_TRACKING_NAMES}
    # self.a_P = {tracking_name: np.diag(a_P[tracking_name]) for tracking_name in NUSCENES_TRACKING_NAMES}
    # self.a_Q = {tracking_name: np.diag(a_Q[tracking_name]) for tracking_name in NUSCENES_TRACKING_NAMES}
    # self.a_R = {tracking_name: np.diag(a_R[tracking_name]) for tracking_name in NUSCENES_TRACKING_NAMES}

    self.m_head_P = np.diag(m_head_P[tracking_name])
    self.m_head_Q = np.diag(m_head_Q[tracking_name])
    self.m_no_head_P = np.diag(m_no_head_P[tracking_name])
    self.m_no_head_Q = np.diag(m_no_head_Q[tracking_name])
    self.m_R = np.diag(m_R[tracking_name])
    self.a_P = np.diag(a_P[tracking_name])
    self.a_Q = np.diag(a_Q[tracking_name])
    self.a_R = np.diag(a_R[tracking_name])