### calibrate

#### load pattern images

??threshold shadow_threshold????
read shadow_threshold = 10
corner_count_x = 7
corner_count_y = 11
corners_width = 21
corners_height = 21.08

---
model_rowCount ??? images_num? or image_set number????
level meaning???
image_set num
---

pattern_set[pattern_list, pattern_list, pattern_list]

std::vector<vector<point2f>> chessboard_corners 
vector<vector<point2f>> projector_corners 
vector<cv::Mat> pattern_list 


pattern_set -> pattern_list
step1:
decode

Out[1]: 

calib = CalibrationData()
calib.cam_K = np.array([[1.34108783e+03, 0.00000000e+00, 7.57647206e+02],
                        [0.00000000e+00, 1.36529204e+03, 5.90305029e+02],
                        [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)
calib.cam_kc = np.array([[0.056331],
                         [-0.14330043],
                         [0.00049911],
                         [0.00587907],
                         [-0.04140769]], dtype=np.float32)
calib.proj_K = np.array([[1.14625724e+03, 0.00000000e+00, 5.18094559e+02],
                         [0.00000000e+00, 2.30183195e+03, 9.03357861e+02],
                         [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)
calib.proj_kc = np.array([[-1.03377180e-01],
                          [1.81623434e-01],
                          [1.17968871e-03],
                          [2.78020677e-04],
                          [-6.04080295e-01]], dtype=np.float32)
calib.R = np.array([[0.91218621, 0.02915611, -0.40873737],
                    [-0.09185049, 0.98663306, -0.13460568],
                    [0.39934922, 0.16032817, 0.90267108]], dtype=np.float32)
calib.T = np.array([[151.97529507],
                    [-53.07685352],
                    [281.33812854]], dtype=np.float32)
calib.is_valid = True
---


### camera

#### K

np.array([[4.61328655e+03, 0.00000000e+00, 1.51430588e+03],
       [0.00000000e+00, 4.61431355e+03, 9.58511402e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)

#### kc
np.array([[-3.15774172e-01],
       [ 7.43419804e+00],
       [ 1.10989491e-04],
       [ 2.70686467e-03],
       [-1.49257829e+02]], dtype = np.float32)

array([[-0.20624257],
       [-0.23927573],
       [ 1.49568526]]), 
array([[-0.51802097],
       [-0.49264543],
       [ 1.44086617]]), 
array([[0.03239998],
       [0.01985874],
       [1.52260968]])], 
array([[ 71.06447578],
       [-75.4761564 ],
       [884.90598094]]), 
array([[120.16756362],
       [-45.93835141],
       [852.83923563]]), 
array([[ 97.31140541],
       [-43.74600524],
       [795.1769794 ]])], 

### projector
#### K
np.array([[9.75347408e+03, 0.00000000e+00, 8.15917991e+02],
       [0.00000000e+00, 8.17254547e+03, 6.02064366e+02],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]], dtype=np.float32)
#### kc
np.array([[-4.77003120e+00],
       [-3.17738270e+03],
       [-1.21848331e-01],
       [ 4.08082144e-02],
       [-6.39263719e+00]], dtype=np.float32)

array([[0.39799633],
       [0.48092941],
       [1.52498437]]), 
array([[-0.6199251 ],
       [-0.41949179],
       [ 1.54153948]]), 
array([[0.42071286],
       [0.64996927],
       [1.45924783]])], 
[array([[  44.46508771],
       [ -92.60454149],
       [2760.89677008]]), 
array([[  70.50120708],
       [ -72.20547298],
       [2593.59104266]]), 
array([[  36.91144936],
       [ -83.72097988],
       [2489.83894303]])]), 

### R, T
#### R
np.array([[ 0.97256075, -0.08113361,  0.21804341],
       [ 0.04049214,  0.98194728,  0.18476993],
       [-0.22909819, -0.17087094,  0.95828865]], dtype=np.float32)

#### t
np.array([[-219.19568903],
       [-185.52863341],
       [1953.00632961]], dtype = np.float32)



array([[-3.65771253e+01, -1.88604780e+03, -5.38646828e+02],
       [ 1.84919996e+03, -1.95908629e+02,  6.35892905e+02],
       [ 1.71562165e+02, -2.30291218e+02, -4.74759955e-02]]), 
array([[ 3.42962639e-07,  1.76804481e-05,  5.83351645e-03],
       [-2.06929833e-05,  2.19177579e-06, -3.59252075e-03],
       [-3.51114632e-03,  5.31068267e-03,  1.00000000e+00]])))


### projector calibration

#### ????????????????????????
> board ?????????
> pattern ?????????????????????

##### board
- board_size: cols=12, rows=9 ????????????
- board_feature_size: cols=11, rows=8 ????????????
- board_distance_world: 20mm ????????????

##### pattern
- pattern_size: cols=10, rows=8 ????????????
- pattern_feature_size: cols=9, rows=7 ????????????
- pattern_distance_pixel: 50pixels ???????????????????????????

##### class calibration
- camera
  - resolution 
  - K, kc
- projector
  - resolution 
  - K, kc
- stereo
  - R, t


#### ??????pose ??????????????????

- projector all on ???1???
- projector all off ???0???
- ???????????????????????????????????????

#### STEP 1

- ??????projector_all_on ??????board feature location ??????????????????
- ????????????????????????????????????????????? ????????? pattern feature location ??????????????????

#### STEP 2

- ??????board ???board_object_points_xyz????????????
- ??????pattern ???projector ??????????????????

#### STEP 3

- camera calibration ?????????????????????board feature location ??? board_object_points_xyz
- ?????????pose??????????????????????????????

#### STEP4

- ????????????pose?????????????????????pattern_feature_location??????pattern_object_points_xyz????????????

#### STEP 5

- ??????pattern_object_points_xyz???pattern_feature_location???projector????????????

#### STEP 6

- stereocamera calibration



