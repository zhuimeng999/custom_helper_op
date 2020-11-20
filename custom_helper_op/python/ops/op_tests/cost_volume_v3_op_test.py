# Copyright 2018 The Sonnet Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Tests for time_two ops."""
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.platform import test
from absl.testing import parameterized
from custom_helper_op import cost_volume_v3, index_initializer
import numpy as np
from tensorflow.python.ops import gradient_checker_v2
from scipy.spatial.transform import Rotation as R
import time

def get_blendedmvs_samples(blendedmvs_data_folder, train_type='train'):
    """ generate data paths for blendedmvs dataset """
    # read data list
    if train_type == 'train':
        proj_info = os.path.join(blendedmvs_data_folder, 'training_list.txt')
    elif train_type == 'valid':
        proj_info = os.path.join(blendedmvs_data_folder, 'validation_list.txt')
    with open(proj_info) as f:
        proj_list = f.read().splitlines()

    # parse all data
    mvs_input_list = []
    for data_name in proj_list:

        dataset_folder = os.path.join(blendedmvs_data_folder, data_name)

        # read cluster
        cluster_path = os.path.join(dataset_folder, 'cams', 'pair.txt')
        with open(cluster_path) as f:
            cluster_lines = f.read().splitlines()
        image_num = int(cluster_lines[0])

        # get per-image info
        for idx in range(0, image_num):

            ref_idx = int(cluster_lines[2 * idx + 1])
            cluster_info = cluster_lines[2 * idx + 2].split()
            total_view_num = int(cluster_info[0])
            if total_view_num < 10:
                continue
            paths = []
            ref_image_path = os.path.join(dataset_folder, 'blended_images', '%08d.jpg' % ref_idx)
            ref_depth_path = os.path.join(dataset_folder, 'rendered_depth_maps', '%08d.pfm' % ref_idx)
            ref_cam_path = os.path.join(dataset_folder, 'cams', '%08d_cam.txt' % ref_idx)
            paths.append(ref_depth_path)
            paths.append(ref_image_path)
            paths.append(ref_cam_path)

            for cidx in range(0, 10):
                view_idx = int(cluster_info[2 * cidx + 1])
                view_image_path = os.path.join(dataset_folder, 'blended_images', '%08d.jpg' % view_idx)
                view_cam_path = os.path.join(dataset_folder, 'cams', '%08d_cam.txt' % view_idx)
                paths.append(view_image_path)
                paths.append(view_cam_path)

            mvs_input_list.append(paths)

    return mvs_input_list

def load_cam(filepath, interval_scale=1):
    """ read camera txt file """
    with open(filepath) as f:
      words = f.read().split()
    # read extrinsic
    cam = np.zeros((2, 4, 4))
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]

    if len(words) == 29:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = FLAGS.max_d
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 30:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = cam[1][3][0] + cam[1][3][1] * cam[1][3][2]
    elif len(words) == 31:
        cam[1][3][0] = words[27]
        cam[1][3][1] = float(words[28]) * interval_scale
        cam[1][3][2] = words[29]
        cam[1][3][3] = words[30]
    else:
        cam[1][3][0] = 0
        cam[1][3][1] = 0
        cam[1][3][2] = 0
        cam[1][3][3] = 0

    return cam

def combine_projection(ref_cam, src_cam, scale):
    ref_R = ref_cam[0, :3, :3]
    ref_T = ref_cam[0, :3, 3:]
    ref_K = ref_cam[1, :3, :3]*scale[:, None]
    src_R = src_cam[0, :3, :3]
    src_T = src_cam[0, :3, 3:]
    src_K = src_cam[1, :3, :3]*scale[:, None]

    ref_K[0, 2] = ref_K[0, 2] + 0.5
    ref_K[1, 2] = ref_K[1, 2] + 0.5
    src_K[0, 2] = src_K[0, 2] + 0.5
    src_K[1, 2] = src_K[1, 2] + 0.5
    with tf.device('cpu'):
        ref_K_inv = tf.linalg.inv(ref_K)

        delta_R = tf.matmul(src_R, tf.transpose(ref_R))
        delta_T = src_T - tf.matmul(delta_R, ref_T)
        R = tf.matmul(src_K, tf.matmul(delta_R, ref_K_inv))
        T = tf.matmul(src_K, delta_T)
    return R, T

def build_sampler_coordinate(R, T, base_plane, offsets, half_centor):
    grid = base_plane + offsets[:, None, None, :]
    base_coordinate = index_initializer(tf.concat([tf.shape(base_plane)[1:3], [3, ]], axis=0),
                                            half_centor=half_centor, dtype=base_plane.dtype)

    coordinate = grid[:, :, :, :, None] * base_coordinate[None, :, :, None, :]
    sample_coodinate = tf.linalg.matvec(R[:, :, None, None, None, :, :], coordinate[:, None, :, :, :, :])
    # sample_coodinate = tf.reduce_sum(R[:, :, None, None, None, :, :] * coordinate[:, None, :, :, :, None, :], axis=-1)
    sample_coodinate = sample_coodinate + T[:, :, None, None, None, :]

    mask = sample_coodinate[..., 2:3] > 0
    tmp = sample_coodinate
    sample_coodinate = tf.where(mask, sample_coodinate[..., :2]/sample_coodinate[..., 2:3], 0)
    if half_centor:
        sample_coodinate = sample_coodinate - tf.constant([0.5, 0.5], dtype=base_plane.dtype)

    return sample_coodinate, grid, tmp


def cost_aggregate_tfa(ref_image, src_images, base_plane, offsets, Rs, Ts, reduce_method="MEAN", half_centor=True):
    image_shape = tf.shape(ref_image)[1:]
    max_d = tf.shape(offsets)[1]
    src_num = tf.shape(src_images)[1]
    sample_coordinate1, grid, coordinate = build_sampler_coordinate(Rs, Ts, base_plane, offsets, half_centor)
    sample_coordinate = tf.reshape(sample_coordinate1, (-1, image_shape[0], image_shape[1], max_d, 2))
    valid_range = (sample_coordinate > 0. ) & (sample_coordinate < tf.reverse(tf.cast(image_shape[:2], ref_image.dtype) - 1, axis=(0,))[None, None, None, None, :])
    valid_range = tf.reduce_all(valid_range, axis=-1, keepdims=True)
    sample_coordinate = tf.where(valid_range, sample_coordinate, -1000)
    maped_feature_volume = tfa.image.resampler(tf.reshape(src_images, (-1, image_shape[0], image_shape[1], image_shape[2])), sample_coordinate)
    maped_feature_volume = tf.reshape(maped_feature_volume,
                                        (-1, src_num, image_shape[0], image_shape[1], max_d, 3, image_shape[2]//3))
    ref_image = tf.reshape(ref_image, (-1, image_shape[0], image_shape[1], 3, image_shape[2]//3))

    cost = tf.reduce_min(tf.reduce_mean(tf.square(ref_image[:, None, :, :, None, :, None, :].numpy() - maped_feature_volume[..., None, :, :].numpy()), axis=-1), axis=(-2, -1))
    if reduce_method == "MEAN":
      cost = tf.reduce_mean(cost, axis=1)
    else:
      cost = tf.reduce_min(cost, axis=1)
    # cost = tf.reshape(cost, (-1, image_shape[0], image_shape[1], max_d, 1))
    return cost, sample_coordinate1, grid, coordinate

class CostVolumeV3Test(test.TestCase, parameterized.TestCase):
  @parameterized.parameters(
    {'BATCH_SIZE':1, 'IMAGE_NUM':1, 'IMAGE_HEIGHT':12, 'IMAGE_WIDTH':16, 'IMAGE_CHANNELS':9, 'IMAGE_DEPTH':5, "reduce_method": "MEAN"},
    {'BATCH_SIZE':1, 'IMAGE_NUM':2, 'IMAGE_HEIGHT':12, 'IMAGE_WIDTH':16, 'IMAGE_CHANNELS':9, 'IMAGE_DEPTH':9, "reduce_method": "MEAN"},
    # {'BATCH_SIZE':1, 'IMAGE_NUM':3, 'IMAGE_HEIGHT':24, 'IMAGE_WIDTH':32, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':9, "reduce_method": "MEAN"},
    {'BATCH_SIZE':2, 'IMAGE_NUM':2, 'IMAGE_HEIGHT':12, 'IMAGE_WIDTH':18, 'IMAGE_CHANNELS':9, 'IMAGE_DEPTH':10, "reduce_method": "MEAN"},
    {'BATCH_SIZE':2, 'IMAGE_NUM':3, 'IMAGE_HEIGHT':12, 'IMAGE_WIDTH':16, 'IMAGE_CHANNELS':9, 'IMAGE_DEPTH':6, "reduce_method": "MEAN"},
    # {'BATCH_SIZE':3, 'IMAGE_NUM':3, 'IMAGE_HEIGHT':24, 'IMAGE_WIDTH':32, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':13, "reduce_method": "MEAN"},
    {'BATCH_SIZE':1, 'IMAGE_NUM':1, 'IMAGE_HEIGHT':12, 'IMAGE_WIDTH':16, 'IMAGE_CHANNELS':9, 'IMAGE_DEPTH':5, "reduce_method": "MIN"},
    {'BATCH_SIZE':1, 'IMAGE_NUM':2, 'IMAGE_HEIGHT':12, 'IMAGE_WIDTH':16, 'IMAGE_CHANNELS':9, 'IMAGE_DEPTH':9, "reduce_method": "MIN"},
    # {'BATCH_SIZE':1, 'IMAGE_NUM':3, 'IMAGE_HEIGHT':24, 'IMAGE_WIDTH':32, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':9, "reduce_method": "MEAN"},
    {'BATCH_SIZE':2, 'IMAGE_NUM':2, 'IMAGE_HEIGHT':12, 'IMAGE_WIDTH':18, 'IMAGE_CHANNELS':9, 'IMAGE_DEPTH':10, "reduce_method": "MIN"},
    {'BATCH_SIZE':2, 'IMAGE_NUM':3, 'IMAGE_HEIGHT':500, 'IMAGE_WIDTH':300, 'IMAGE_CHANNELS':9, 'IMAGE_DEPTH':6, "reduce_method": "MIN"},
    # {'BATCH_SIZE':3, 'IMAGE_NUM':3, 'IMAGE_HEIGHT':24, 'IMAGE_WIDTH':32, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':13, "reduce_method": "MEAN"},
  )
  def testCostVolumeV3Simple(self, BATCH_SIZE = 2, IMAGE_NUM = 2, IMAGE_HEIGHT = 5, IMAGE_WIDTH = 5, IMAGE_CHANNELS = 3, IMAGE_DEPTH = 4, reduce_method= "MEAN", half_centor=True):
    batch_ref_image = tf.random.uniform([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS], dtype=tf.float64)*10
    batch_src_images = tf.random.uniform([BATCH_SIZE, IMAGE_NUM, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS], dtype=batch_ref_image.dtype)*10
    batch_ref_depth = (tf.random.uniform([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 1], dtype=batch_src_images.dtype) + 2)*10
    batch_offsets = tf.random.uniform([BATCH_SIZE, IMAGE_DEPTH], dtype=batch_src_images.dtype)*4 - 2
    batch_Rs = tf.cast(np.tile(np.diagflat([1., 1., 1.])[None, None, ...], [BATCH_SIZE, IMAGE_NUM, 1, 1]), batch_src_images.dtype)
    batch_Ts = tf.cast(np.tile(np.array([0., 0., 0.])[None, None, ...], [BATCH_SIZE, IMAGE_NUM, 1]), batch_src_images.dtype)


    cost, cost_mask = cost_volume_v3(batch_ref_image, batch_src_images, batch_ref_depth, batch_offsets, batch_Rs, batch_Ts, reduce_method=reduce_method, groups=3, half_centor=half_centor)
    if reduce_method == "MEAN":
        cost = tf.where(cost_mask >= IMAGE_NUM, cost, 0.)
    else:
        cost = tf.where(cost_mask[..., 0:1] >= 0, cost, 0.)

    cost_tfa, sample_coordinate, grid, coordinate = cost_aggregate_tfa(batch_ref_image, batch_src_images, batch_ref_depth, batch_offsets, batch_Rs, batch_Ts, reduce_method=reduce_method, half_centor=half_centor)
    if reduce_method == "MEAN":
        cost_tfa = tf.where(cost_mask >= IMAGE_NUM, cost_tfa[..., None], 0.)
    else:
        cost_tfa = tf.where(cost_mask[..., 0:1] >= 0, cost_tfa[..., None], 0.)

    self.assertAllClose(cost_tfa.numpy()[:, 1:-1, 1:-1, :] , cost.numpy()[:, 1:-1, 1:-1, :], rtol=1e-5)

    print("1: ",np.max(cost_tfa.numpy()), np.max(cost.numpy()))

    batch_Ts = np.tile(np.array([3., 3., 3.])[None, None, ...], [BATCH_SIZE, IMAGE_NUM, 1])
    cost, cost_mask = cost_volume_v3(batch_ref_image, batch_src_images, batch_ref_depth, batch_offsets, batch_Rs, batch_Ts, reduce_method=reduce_method, groups=3, half_centor=half_centor)
    if reduce_method == "MEAN":
        cost = tf.where(cost_mask >= IMAGE_NUM, cost, 0.)
    else:
        cost = tf.where(cost_mask[..., 0:1] >= 0, cost, 0.)

    cost_tfa, sample_coordinate, grid, coordinate = cost_aggregate_tfa(batch_ref_image, batch_src_images, batch_ref_depth, batch_offsets, batch_Rs, batch_Ts, reduce_method=reduce_method, half_centor=half_centor)
    if reduce_method == "MEAN":
        cost_tfa = tf.where(cost_mask >= IMAGE_NUM, cost_tfa[..., None], 0.)
    else:
        cost_tfa = tf.where(cost_mask[..., 0:1] >= 0, cost_tfa[..., None], 0.)

    self.assertAllClose(cost_tfa.numpy()[:, 1:-1, 1:-1, :] , cost.numpy()[:, 1:-1, 1:-1, :], rtol=1e-5)

    print("2: ",np.max(cost_tfa.numpy()), np.max(cost.numpy()))


    batch_Rs = np.tile(R.from_rotvec(np.pi/2 * np.array([0, 0, 1])).as_matrix()[None, None, ...], [BATCH_SIZE, IMAGE_NUM, 1, 1])
    cost, cost_mask = cost_volume_v3(batch_ref_image, batch_src_images, batch_ref_depth, batch_offsets, batch_Rs, batch_Ts, reduce_method=reduce_method, groups=3, half_centor=half_centor)
    if reduce_method == "MEAN":
        cost = tf.where(cost_mask >= IMAGE_NUM, cost, 0.)
    else:
        cost = tf.where(cost_mask[..., 0:1] >= 0, cost, 0.)

    cost_tfa, sample_coordinate, grid, coordinate = cost_aggregate_tfa(batch_ref_image, batch_src_images, batch_ref_depth, batch_offsets, batch_Rs, batch_Ts, reduce_method=reduce_method, half_centor=half_centor)
    if reduce_method == "MEAN":
        cost_tfa = tf.where(cost_mask >= IMAGE_NUM, cost_tfa[..., None], 0.)
    else:
        cost_tfa = tf.where(cost_mask[..., 0:1] >= 0, cost_tfa[..., None], 0.)

    self.assertAllClose(cost_tfa.numpy()[:, 1:-1, 1:-1, :] , cost.numpy()[:, 1:-1, 1:-1, :], rtol=1e-5)

    print("3: ",np.max(cost_tfa.numpy()), np.max(cost.numpy()))

    batch_Rs = np.tile(R.from_rotvec(np.pi/8 * np.array([0, 1, 0])).as_matrix()[None, None, ...], [BATCH_SIZE, IMAGE_NUM, 1, 1])
    cost, cost_mask = cost_volume_v3(batch_ref_image, batch_src_images, batch_ref_depth, batch_offsets, batch_Rs, batch_Ts, reduce_method=reduce_method, groups=3, half_centor=half_centor)
    if reduce_method == "MEAN":
        cost = tf.where(cost_mask >= IMAGE_NUM, cost, 0.)
    else:
        cost = tf.where(cost_mask[..., 0:1] >= 0, cost, 0.)

    cost_tfa, sample_coordinate, grid, coordinate = cost_aggregate_tfa(batch_ref_image, batch_src_images, batch_ref_depth, batch_offsets, batch_Rs, batch_Ts, reduce_method=reduce_method, half_centor=half_centor)
    if reduce_method == "MEAN":
        cost_tfa = tf.where(cost_mask >= IMAGE_NUM, cost_tfa[..., None], 0.)
    else:
        cost_tfa = tf.where(cost_mask[..., 0:1] >= 0, cost_tfa[..., None], 0.)

    self.assertAllClose(cost_tfa.numpy()[:, 1:-1, 1:-1, :] , cost.numpy()[:, 1:-1, 1:-1, :], rtol=1e-5)

    print("4: ",np.max(cost_tfa.numpy()), np.max(cost.numpy()))

#   @test_util.parameterized.parameters(
#     {'BATCH_SIZE':1, 'IMAGE_NUM':1, 'IMAGE_HEIGHT':512, 'IMAGE_WIDTH':640, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':5, "reduce_method": "MEAN"},
#     {'BATCH_SIZE':1, 'IMAGE_NUM':2, 'IMAGE_HEIGHT':512, 'IMAGE_WIDTH':640, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':9, "reduce_method": "MEAN"},
#     {'BATCH_SIZE':1, 'IMAGE_NUM':3, 'IMAGE_HEIGHT':512, 'IMAGE_WIDTH':640, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':9, "reduce_method": "MEAN"},
#     # {'BATCH_SIZE':2, 'IMAGE_NUM':1, 'IMAGE_HEIGHT':512, 'IMAGE_WIDTH':640, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':10, "reduce_method": "MEAN"},
#     # {'BATCH_SIZE':2, 'IMAGE_NUM':2, 'IMAGE_HEIGHT':512, 'IMAGE_WIDTH':640, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':11, "reduce_method": "MEAN"},
#     # {'BATCH_SIZE':3, 'IMAGE_NUM':3, 'IMAGE_HEIGHT':256, 'IMAGE_WIDTH':320, 'IMAGE_CHANNELS':13, 'IMAGE_DEPTH':4, "reduce_method": "MEAN"},

#     {'BATCH_SIZE':1, 'IMAGE_NUM':1, 'IMAGE_HEIGHT':512, 'IMAGE_WIDTH':640, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':5, "reduce_method": "MIN"},
#     {'BATCH_SIZE':1, 'IMAGE_NUM':2, 'IMAGE_HEIGHT':512, 'IMAGE_WIDTH':640, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':9, "reduce_method": "MIN"},
#     {'BATCH_SIZE':1, 'IMAGE_NUM':3, 'IMAGE_HEIGHT':512, 'IMAGE_WIDTH':640, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':9, "reduce_method": "MIN"},
#     # {'BATCH_SIZE':2, 'IMAGE_NUM':1, 'IMAGE_HEIGHT':512, 'IMAGE_WIDTH':640, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':10, "reduce_method": "MIN"},
#     # {'BATCH_SIZE':2, 'IMAGE_NUM':2, 'IMAGE_HEIGHT':512, 'IMAGE_WIDTH':640, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':11, "reduce_method": "MIN"},
#     # {'BATCH_SIZE':3, 'IMAGE_NUM':3, 'IMAGE_HEIGHT':256, 'IMAGE_WIDTH':320, 'IMAGE_CHANNELS':13, 'IMAGE_DEPTH':4, "reduce_method": "MIN"}
#   )
#   def testCostAggregate(self, BATCH_SIZE = 1, IMAGE_NUM = 2, IMAGE_HEIGHT = 10, IMAGE_WIDTH = 20, IMAGE_CHANNELS = 32, IMAGE_DEPTH = 256, reduce_method= "MEAN"):
#     mvs_input_list = get_blendedmvs_samples("/home/lucius/data/datasets/mvsnet/dataset_low_res")
#     np.random.shuffle(mvs_input_list)
#     for i in range(10):
#       batch_ref_depth = []
#       batch_ref_image = []
#       batch_src_images = []
#       batch_offsets = []
#       batch_Rs = []
#       batch_Ts = []
#       for b in range(BATCH_SIZE):
#         ref_depth = cv2.imread(mvs_input_list[i*BATCH_SIZE + b][0], cv2.IMREAD_UNCHANGED)
#         ref_image = cv2.imread(mvs_input_list[i*BATCH_SIZE + b][1], cv2.IMREAD_UNCHANGED)
#         ref_cam = load_cam(mvs_input_list[i*BATCH_SIZE + b][2])
#         internal = (ref_cam[1][3][3] - ref_cam[1][3][0])/(IMAGE_DEPTH - 1)
#         batch_offsets.append(0.5*internal*tf.linspace(-IMAGE_DEPTH/2, IMAGE_DEPTH/2 + 1, IMAGE_DEPTH) )

#         scale = np.array([IMAGE_HEIGHT, IMAGE_WIDTH, 3.], dtype=np.float)/ref_image.shape
#         src_images = []
#         src_Rs = []
#         src_Ts = []
#         for n in range(IMAGE_NUM):
#           src_images.append(tf.image.resize(cv2.imread(mvs_input_list[i*BATCH_SIZE + b][2*n + 3], cv2.IMREAD_UNCHANGED)/256., (IMAGE_HEIGHT, IMAGE_WIDTH), method='area' ))
#           src_cam = load_cam(mvs_input_list[i*BATCH_SIZE + b][2*n + 4])
#           R, T = combine_projection(ref_cam, src_cam, scale)
#           src_Rs.append(R)
#           src_Ts.append(T)

#         batch_ref_depth.append(tf.image.resize(ref_depth[:, :, None], (IMAGE_HEIGHT, IMAGE_WIDTH), method='bilinear'))
#         batch_ref_image.append(tf.image.resize(ref_image/256., (IMAGE_HEIGHT, IMAGE_WIDTH), method='area'))
#         batch_src_images.append(tf.stack(src_images, axis=0))
#         batch_Rs.append(tf.stack(src_Rs, axis=0))
#         batch_Ts.append(tf.stack(src_Ts, axis=0))


#       batch_ref_depth = tf.cast(tf.stack(batch_ref_depth, axis=0), tf.float64)
#       batch_ref_image = tf.cast(tf.stack(batch_ref_image, axis=0), tf.float64)*10 + 100.
#       batch_src_images = tf.cast(tf.stack(batch_src_images, axis=0), tf.float64)*10 + 100.
#       batch_offsets = tf.cast(tf.stack(batch_offsets, axis=0), tf.float64)
#       batch_Rs = tf.cast(tf.stack(batch_Rs, axis=0), tf.float64)
#       batch_Ts = tf.squeeze(tf.cast(tf.stack(batch_Ts, axis=0), tf.float64), axis=-1)

#       start = time.time()
#       cost, cost_mask = cost_aggregate(batch_ref_image, batch_src_images, batch_ref_depth, batch_offsets, batch_Rs, batch_Ts, reduce_method=reduce_method)
#       if reduce_method == "MEAN":
#         cost = tf.where(cost_mask >= IMAGE_NUM, cost, 0.)
#       base_time = time.time() - start
#       start = time.time()
#       cost_tfa, sample_coordinate, grid, coordinate = cost_aggregate_tfa(batch_ref_image, batch_src_images, batch_ref_depth, batch_offsets, batch_Rs, batch_Ts, reduce_method=reduce_method)
#       if reduce_method == "MEAN":
#         cost_tfa = tf.where(cost_mask >= IMAGE_NUM, cost_tfa, 0.)
#       else:
#         cost_tfa = tf.where(cost_mask >= 0, cost_tfa, 0.)
#       tfa_time = time.time() - start

#       print(np.mean(cost), " base_time: ", base_time/1000, " tfa_time: ", tfa_time/1000)
#       np.testing.assert_allclose(cost_tfa.numpy() , cost.numpy(), rtol=1e-2, atol=1e-5)


#   @test_util.parameterized.parameters(
#     {'BATCH_SIZE':1, 'IMAGE_NUM':1, 'IMAGE_HEIGHT':24, 'IMAGE_WIDTH':32, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':5, "reduce_method": "MEAN"},
#     {'BATCH_SIZE':1, 'IMAGE_NUM':2, 'IMAGE_HEIGHT':12, 'IMAGE_WIDTH':16, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':5, "reduce_method": "MEAN"},
#     # {'BATCH_SIZE':1, 'IMAGE_NUM':3, 'IMAGE_HEIGHT':12, 'IMAGE_WIDTH':16, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':4, "reduce_method": "MEAN"},
#     # {'BATCH_SIZE':2, 'IMAGE_NUM':1, 'IMAGE_HEIGHT':24, 'IMAGE_WIDTH':32, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':5, "reduce_method": "MEAN"},
#     # {'BATCH_SIZE':2, 'IMAGE_NUM':2, 'IMAGE_HEIGHT':12, 'IMAGE_WIDTH':16, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':4, "reduce_method": "MEAN"},
#     # {'BATCH_SIZE':3, 'IMAGE_NUM':3, 'IMAGE_HEIGHT':12, 'IMAGE_WIDTH':16, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':3, "reduce_method": "MEAN"}
#   )
#   def testCostAggregateGradDataset(self, BATCH_SIZE = 1, IMAGE_NUM = 2, IMAGE_HEIGHT = 10, IMAGE_WIDTH = 20, IMAGE_CHANNELS = 32, IMAGE_DEPTH = 256, reduce_method= "MEAN"):
#     @tf.function
#     def test_check(*args):
#         cost, *_ = cost_aggregate(*args, reduce_method=reduce_method)
#         return tf.reduce_mean(cost)

#     mvs_input_list = get_blendedmvs_samples("/home/lucius/data/datasets/mvsnet/dataset_low_res")
#     np.random.shuffle(mvs_input_list)
#     for i in range(10):
#       batch_ref_depth = []
#       batch_ref_image = []
#       batch_src_images = []
#       batch_offsets = []
#       batch_Rs = []
#       batch_Ts = []
#       for b in range(BATCH_SIZE):
#         ref_depth = cv2.imread(mvs_input_list[i*BATCH_SIZE + b][0], cv2.IMREAD_UNCHANGED)
#         ref_image = cv2.imread(mvs_input_list[i*BATCH_SIZE + b][1], cv2.IMREAD_UNCHANGED)
#         ref_cam = load_cam(mvs_input_list[i*BATCH_SIZE + b][2])
#         internal = (ref_cam[1][3][3] - ref_cam[1][3][0])/(IMAGE_DEPTH - 1)
#         batch_offsets.append(0.5*internal*tf.linspace(-IMAGE_DEPTH/2, IMAGE_DEPTH/2 + 1, IMAGE_DEPTH) )

#         scale = np.array([IMAGE_HEIGHT, IMAGE_WIDTH, 3.], dtype=np.float)/ref_image.shape
#         src_images = []
#         src_Rs = []
#         src_Ts = []
#         for n in range(IMAGE_NUM):
#           src_images.append(tf.image.resize(cv2.imread(mvs_input_list[i*BATCH_SIZE + b][2*n + 3], cv2.IMREAD_UNCHANGED)/256., (IMAGE_HEIGHT, IMAGE_WIDTH), method='area' ))
#           src_cam = load_cam(mvs_input_list[i*BATCH_SIZE + b][2*n + 4])
#           R, T = combine_projection(ref_cam, src_cam, scale)
#           src_Rs.append(R)
#           src_Ts.append(T)

#         batch_ref_depth.append(tf.image.resize(ref_depth[:, :, None], (IMAGE_HEIGHT, IMAGE_WIDTH), method='bilinear'))
#         batch_ref_image.append(tf.image.resize(ref_image/256., (IMAGE_HEIGHT, IMAGE_WIDTH), method='area'))
#         batch_src_images.append(tf.stack(src_images, axis=0))
#         batch_Rs.append(tf.stack(src_Rs, axis=0))
#         batch_Ts.append(tf.stack(src_Ts, axis=0))


#       batch_ref_depth = tf.cast(tf.stack(batch_ref_depth, axis=0), tf.float64)
#       batch_ref_image = tf.cast(tf.stack(batch_ref_image, axis=0), tf.float64)
#       batch_src_images = tf.cast(tf.stack(batch_src_images, axis=0), tf.float64)
#       batch_offsets = tf.cast(tf.stack(batch_offsets, axis=0), tf.float64)
#       batch_Rs = tf.cast(tf.stack(batch_Rs, axis=0), tf.float64)
#       batch_Ts = tf.squeeze(tf.cast(tf.stack(batch_Ts, axis=0), tf.float64), axis=-1)

#       theoretical, numerical = tf.test.compute_gradient(test_check, [batch_ref_image, batch_src_images, batch_ref_depth, batch_offsets, batch_Rs, batch_Ts])

#       np.testing.assert_allclose(theoretical[0] , numerical[0], rtol=5e-5, atol=1e-6)
#       np.testing.assert_allclose(theoretical[1] , numerical[1], rtol=5e-5, atol=1e-6)
#       np.testing.assert_allclose(theoretical[2] , numerical[2], rtol=5e-5, atol=1e-6)

  @parameterized.parameters(
    # {'BATCH_SIZE':1, 'IMAGE_NUM':1, 'IMAGE_HEIGHT':12, 'IMAGE_WIDTH':16, 'IMAGE_CHANNELS':9, 'IMAGE_DEPTH':5, "reduce_method": "MEAN"},
    # {'BATCH_SIZE':1, 'IMAGE_NUM':2, 'IMAGE_HEIGHT':12, 'IMAGE_WIDTH':16, 'IMAGE_CHANNELS':9, 'IMAGE_DEPTH':9, "reduce_method": "MEAN"},
    # # {'BATCH_SIZE':1, 'IMAGE_NUM':3, 'IMAGE_HEIGHT':24, 'IMAGE_WIDTH':32, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':9, "reduce_method": "MEAN"},
    # {'BATCH_SIZE':2, 'IMAGE_NUM':2, 'IMAGE_HEIGHT':12, 'IMAGE_WIDTH':18, 'IMAGE_CHANNELS':9, 'IMAGE_DEPTH':10, "reduce_method": "MEAN"},
    # {'BATCH_SIZE':2, 'IMAGE_NUM':3, 'IMAGE_HEIGHT':12, 'IMAGE_WIDTH':16, 'IMAGE_CHANNELS':9, 'IMAGE_DEPTH':6, "reduce_method": "MEAN"},
    # {'BATCH_SIZE':3, 'IMAGE_NUM':3, 'IMAGE_HEIGHT':24, 'IMAGE_WIDTH':32, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':13, "reduce_method": "MEAN"},
    {'BATCH_SIZE':1, 'IMAGE_NUM':1, 'IMAGE_HEIGHT':12, 'IMAGE_WIDTH':16, 'IMAGE_CHANNELS':9, 'IMAGE_DEPTH':5, "reduce_method": "MIN"},
    {'BATCH_SIZE':1, 'IMAGE_NUM':2, 'IMAGE_HEIGHT':12, 'IMAGE_WIDTH':16, 'IMAGE_CHANNELS':9, 'IMAGE_DEPTH':9, "reduce_method": "MIN"},
    # {'BATCH_SIZE':1, 'IMAGE_NUM':3, 'IMAGE_HEIGHT':24, 'IMAGE_WIDTH':32, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':9, "reduce_method": "MEAN"},
    {'BATCH_SIZE':2, 'IMAGE_NUM':2, 'IMAGE_HEIGHT':12, 'IMAGE_WIDTH':18, 'IMAGE_CHANNELS':9, 'IMAGE_DEPTH':10, "reduce_method": "MIN"},
    {'BATCH_SIZE':2, 'IMAGE_NUM':3, 'IMAGE_HEIGHT':12, 'IMAGE_WIDTH':16, 'IMAGE_CHANNELS':9, 'IMAGE_DEPTH':6, "reduce_method": "MIN"},
    # {'BATCH_SIZE':3, 'IMAGE_NUM':3, 'IMAGE_HEIGHT':24, 'IMAGE_WIDTH':32, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':13, "reduce_method": "MEAN"},
  )
  def testCostVolumeV3Grad(self, BATCH_SIZE = 2, IMAGE_NUM = 2, IMAGE_HEIGHT = 5, IMAGE_WIDTH = 5, IMAGE_CHANNELS = 3, IMAGE_DEPTH = 4, reduce_method= "MEAN"):
    batch_ref_image = tf.random.uniform([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS], dtype=tf.float64)*10
    batch_src_images = tf.random.uniform([BATCH_SIZE, IMAGE_NUM, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS], dtype=batch_ref_image.dtype)*10
    batch_ref_depth = (tf.random.uniform([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 1], dtype=batch_src_images.dtype) + 2)*10
    batch_offsets = tf.random.uniform([BATCH_SIZE, IMAGE_DEPTH], dtype=batch_src_images.dtype)*4 - 2
    batch_Rs = tf.cast(np.tile(np.diagflat([1., 1., 1.])[None, None, ...], [BATCH_SIZE, IMAGE_NUM, 1, 1]), batch_src_images.dtype)
    batch_Ts = tf.cast(np.tile(np.array([0., 0., 0.])[None, None, ...], [BATCH_SIZE, IMAGE_NUM, 1]), batch_src_images.dtype)

    @tf.function
    def test_check(*args):
        cost, cost_mask = cost_volume_v3(*args, reduce_method=reduce_method, groups=3)
        # if reduce_method == "MEAN":
        #     cost = tf.where(cost_mask > 0, cost, 0.)
        # else:
        #     cost = tf.where(cost_mask[..., 0:1] >= 0, cost, 0.)
        return tf.reduce_sum(cost)

    theoretical, numerical = tf.test.compute_gradient(test_check, [batch_ref_image, batch_src_images, batch_ref_depth, batch_offsets, batch_Rs, batch_Ts])

    test_grad_index = 0
    idx = np.argmax(np.abs(theoretical[test_grad_index] - numerical[test_grad_index]))
    b, h, w, c = idx//(IMAGE_HEIGHT*IMAGE_WIDTH*IMAGE_CHANNELS), (idx//(IMAGE_WIDTH*IMAGE_CHANNELS))%IMAGE_HEIGHT, (idx//IMAGE_CHANNELS)%IMAGE_WIDTH, idx%IMAGE_CHANNELS
    print(b, h, w, c, batch_ref_image[b, h, w, c], theoretical[test_grad_index][0, idx], numerical[test_grad_index][0, idx])

    self.assertAllClose(theoretical[0] , numerical[0], rtol=5e-5)
    self.assertAllClose(theoretical[1] , numerical[1], rtol=5e-5)
    
    # np.testing.assert_allclose(theoretical[2] , numerical[2], rtol=5e-5, atol=1e-6)
    print("1: ",np.max(theoretical[0]), np.max(theoretical[1]), np.max(theoretical[2]))

    batch_Ts = np.tile(np.array([3., 3., 3.])[None, None, ...], [BATCH_SIZE, IMAGE_NUM, 1])
    theoretical, numerical = tf.test.compute_gradient(test_check, [batch_ref_image, batch_src_images, batch_ref_depth, batch_offsets, batch_Rs, batch_Ts])



    self.assertAllClose(theoretical[0] , numerical[0], rtol=5e-5)
    self.assertAllClose(theoretical[1] , numerical[1], rtol=5e-5)
    self.assertAllClose(theoretical[2] , numerical[2], rtol=5e-5)
    print("2: ",np.max(theoretical[0]), np.max(theoretical[1]), np.max(theoretical[2]))

    # batch_Rs = np.tile(R.from_rotvec(np.pi/2 * np.array([0, 0, 1])).as_matrix()[None, None, ...], [BATCH_SIZE, IMAGE_NUM, 1, 1])
    # theoretical, numerical = tf.test.compute_gradient(test_check, [batch_ref_image, batch_src_images, batch_ref_depth, batch_offsets, batch_Rs, batch_Ts])
    # np.testing.assert_allclose(theoretical[0] , numerical[0], rtol=5e-5, atol=1e-6)
    # np.testing.assert_allclose(theoretical[1] , numerical[1], rtol=5e-5, atol=1e-6)
    # np.testing.assert_allclose(theoretical[2] , numerical[2], rtol=5e-5, atol=1e-6)
    # print("3: ",np.max(theoretical[0]), np.max(theoretical[1]), np.max(theoretical[2]))

    batch_Rs = np.tile(R.from_rotvec(np.pi/8 * np.array([0, 1, 0])).as_matrix()[None, None, ...], [BATCH_SIZE, IMAGE_NUM, 1, 1])
    theoretical, numerical = tf.test.compute_gradient(test_check, [batch_ref_image, batch_src_images, batch_ref_depth, batch_offsets, batch_Rs, batch_Ts])
    self.assertAllClose(theoretical[0] , numerical[0], rtol=5e-5)
    self.assertAllClose(theoretical[1] , numerical[1], rtol=5e-5)
    self.assertAllClose(theoretical[2] , numerical[2], rtol=5e-5)
    print("4: ",np.max(theoretical[0]), np.max(theoretical[1]), np.max(theoretical[2]))

if __name__ == '__main__':
  test.main()
