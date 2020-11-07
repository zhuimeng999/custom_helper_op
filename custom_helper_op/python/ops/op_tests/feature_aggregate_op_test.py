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
from custom_helper_op import feature_aggregate, index_initializer
import numpy as np
from tensorflow.python.ops import gradient_checker_v2
import time
from scipy.spatial.transform import Rotation as R

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
    sample_coodinate = tf.where(mask, sample_coodinate[..., :2]/sample_coodinate[..., 2:3], 0)
    if half_centor:
        sample_coodinate = sample_coodinate - tf.constant([0.5, 0.5], dtype=base_plane.dtype)

    return sample_coodinate, grid


def cost_aggregate_tfa(src_images, base_plane, offsets, Rs, Ts, reduce_method="MEAN", half_centor=True):
    src_image_shape = tf.shape(src_images)
    image_shape = tf.shape(base_plane)[1:3]
    max_d = tf.shape(offsets)[1]
    sample_coordinate1, grid = build_sampler_coordinate(Rs, Ts, base_plane, offsets, half_centor)
    sample_coordinate = tf.reshape(sample_coordinate1, (-1, image_shape[0], image_shape[1], max_d, 2))
    # valid_range = (sample_coordinate >= 0. ) & (sample_coordinate < tf.reverse(tf.cast(image_shape, base_plane.dtype) - 1, axis=(0,))[None, None, None, None, :])
    # valid_range = tf.reduce_all(valid_range, axis=-1, keepdims=True)
    # sample_coordinate = tf.where(valid_range, sample_coordinate, -1000)
    maped_feature_volume = tfa.image.resampler(tf.reshape(src_images, (-1, src_image_shape[2], src_image_shape[3], src_image_shape[4])), sample_coordinate)
    maped_feature_volume = tf.reshape(maped_feature_volume,
                                        (-1, src_image_shape[1], image_shape[0], image_shape[1], max_d, src_image_shape[4]))
    return maped_feature_volume

class FeatureAggregateTest(test.TestCase, parameterized.TestCase):
  @parameterized.parameters(
    (1, 2, 70, 80, 3, 5, (7, 8)),
    # {'BATCH_SIZE':3, 'IMAGE_NUM':3, 'IMAGE_HEIGHT':24, 'IMAGE_WIDTH':32, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':13, "reduce_method": "MEAN"},
  )
  def testForward(self, BATCH_SIZE = 2, IMAGE_NUM = 2, IMAGE_HEIGHT = 5, IMAGE_WIDTH = 5, IMAGE_CHANNELS = 3, IMAGE_DEPTH = 4, output_shape=(5, 5), half_centor=True):
    src_images = tf.random.uniform([BATCH_SIZE, IMAGE_NUM, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS], dtype=tf.float64)*10
    base_plane = (tf.random.uniform([BATCH_SIZE, output_shape[0], output_shape[1], 1], dtype=src_images.dtype) + 2)*10
    offsets = tf.random.uniform([BATCH_SIZE, IMAGE_DEPTH], dtype=src_images.dtype)*4 - 2
    Rs = tf.cast(np.tile(np.diagflat([1., 1., 1.])[None, None, ...], [BATCH_SIZE, IMAGE_NUM, 1, 1]), src_images.dtype)
    Ts = tf.cast(np.tile(np.array([20., 20., 10.])[None, None, ...], [BATCH_SIZE, IMAGE_NUM, 1]), src_images.dtype)


    mapped_feature, mapped_mask = feature_aggregate(src_images, base_plane, offsets, Rs, Ts, half_centor=half_centor)


    mapped_feature_tfa = cost_aggregate_tfa(src_images, base_plane, offsets, Rs, Ts, half_centor=half_centor)


    self.assertAllClose(mapped_feature, tf.where(mapped_mask == 1, tf.transpose(mapped_feature_tfa, [0, 2, 3, 4, 1, 5]), 0))

    print("1: ",np.max(mapped_feature.numpy()), np.max(mapped_feature_tfa.numpy()))

    Ts = tf.cast(np.tile(np.array([20., 20., 3.])[None, None, ...], [BATCH_SIZE, IMAGE_NUM, 1]), src_images.dtype)
    mapped_feature, mapped_mask = feature_aggregate(src_images, base_plane, offsets, Rs, Ts, half_centor=half_centor)


    mapped_feature_tfa = cost_aggregate_tfa(src_images, base_plane, offsets, Rs, Ts, half_centor=half_centor)

    self.assertAllClose(mapped_feature, tf.where(mapped_mask == 1, tf.transpose(mapped_feature_tfa, [0, 2, 3, 4, 1, 5]), 0))

    print("2: ",np.max(mapped_feature.numpy()), np.max(mapped_feature_tfa.numpy()))


    Rs = tf.cast(np.tile(R.from_rotvec(np.pi/2 * np.array([0, 0, 1])).as_matrix()[None, None, ...], [BATCH_SIZE, IMAGE_NUM, 1, 1]), src_images.dtype)
    mapped_feature, mapped_mask = feature_aggregate(src_images, base_plane, offsets, Rs, Ts, half_centor=half_centor)


    mapped_feature_tfa = cost_aggregate_tfa(src_images, base_plane, offsets, Rs, Ts, half_centor=half_centor)

    self.assertAllClose(mapped_feature, tf.where(mapped_mask == 1, tf.transpose(mapped_feature_tfa, [0, 2, 3, 4, 1, 5]), 0))

    print("3: ",np.max(mapped_feature.numpy()), np.max(mapped_feature_tfa.numpy()))

    Rs = tf.cast(np.tile(R.from_rotvec(np.pi/8 * np.array([0, 1, 0])).as_matrix()[None, None, ...], [BATCH_SIZE, IMAGE_NUM, 1, 1]), src_images.dtype)
    mapped_feature, mapped_mask = feature_aggregate(src_images, base_plane, offsets, Rs, Ts, half_centor=half_centor)


    mapped_feature_tfa = cost_aggregate_tfa(src_images, base_plane, offsets, Rs, Ts, half_centor=half_centor)

    self.assertAllClose(mapped_feature, tf.where(mapped_mask == 1, tf.transpose(mapped_feature_tfa, [0, 2, 3, 4, 1, 5]), 0))
    print("4: ",np.max(mapped_feature.numpy()), np.max(mapped_feature_tfa.numpy()))


  @parameterized.parameters(
    (1, 2, 70, 80, 3, 5, (7, 8)),
    # {'BATCH_SIZE':3, 'IMAGE_NUM':3, 'IMAGE_HEIGHT':24, 'IMAGE_WIDTH':32, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':13, "reduce_method": "MEAN"},
  )
  def testBackforward(self, BATCH_SIZE = 2, IMAGE_NUM = 2, IMAGE_HEIGHT = 5, IMAGE_WIDTH = 5, IMAGE_CHANNELS = 3, IMAGE_DEPTH = 4, output_shape=(5, 5), half_centor=True):
    src_images = tf.random.uniform([BATCH_SIZE, IMAGE_NUM, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS], dtype=tf.float64)*10
    base_plane = (tf.random.uniform([BATCH_SIZE, output_shape[0], output_shape[1], 1], dtype=src_images.dtype) + 2)*10
    offsets = tf.random.uniform([BATCH_SIZE, IMAGE_DEPTH], dtype=src_images.dtype)*4 - 2
    Rs = tf.cast(np.tile(np.diagflat([1., 1., 1.])[None, None, ...], [BATCH_SIZE, IMAGE_NUM, 1, 1]), src_images.dtype)
    Ts = tf.cast(np.tile(np.array([20., 20., 10.])[None, None, ...], [BATCH_SIZE, IMAGE_NUM, 1]), src_images.dtype)

    @tf.function
    def test_check(*args):
        cost, *_ = feature_aggregate(*args, half_centor=half_centor)
        return tf.reduce_mean(cost)

    theoretical, numerical = tf.test.compute_gradient(test_check, [src_images, base_plane, offsets, Rs, Ts])
    # idx = np.argmax(theoretical[2] - numerical[2])
    # b, h, w = idx//(IMAGE_HEIGHT*IMAGE_WIDTH), (idx//IMAGE_WIDTH)%IMAGE_HEIGHT, idx%IMAGE_WIDTH
    # print(b, h, w, batch_ref_depth[b, h, w, 0], theoretical[2][0, idx], numerical[2][0, idx])

    self.assertAllClose(theoretical[0] , numerical[0])
    self.assertAllClose(theoretical[1] , numerical[1])
    # np.testing.assert_allclose(theoretical[2] , numerical[2], rtol=5e-5, atol=1e-6)
    print("1: ",np.max(theoretical[0]), np.max(theoretical[1]), np.max(theoretical[2]))

    Ts = tf.cast(np.tile(np.array([3., 3., 3.])[None, None, ...], [BATCH_SIZE, IMAGE_NUM, 1]), src_images.dtype)
    theoretical, numerical = tf.test.compute_gradient(test_check, [src_images, base_plane, offsets, Rs, Ts])

    self.assertAllClose(theoretical[0] , numerical[0])
    self.assertAllClose(theoretical[1] , numerical[1])
    # np.testing.assert_allclose(theoretical[2] , numerical[2], rtol=5e-5, atol=1e-6)
    print("2: ",np.max(theoretical[0]), np.max(theoretical[1]), np.max(theoretical[2]))

    Rs = tf.cast(np.tile(R.from_rotvec(np.pi/2 * np.array([0, 0, 1])).as_matrix()[None, None, ...], [BATCH_SIZE, IMAGE_NUM, 1, 1]), src_images.dtype)
    theoretical, numerical = tf.test.compute_gradient(test_check, [src_images, base_plane, offsets, Rs, Ts])
    self.assertAllClose(theoretical[0] , numerical[0])
    self.assertAllClose(theoretical[1] , numerical[1])
    # np.testing.assert_allclose(theoretical[2] , numerical[2], rtol=5e-5, atol=1e-6)
    print("3: ",np.max(theoretical[0]), np.max(theoretical[1]), np.max(theoretical[2]))

    Rs = tf.cast(np.tile(R.from_rotvec(np.pi/8 * np.array([0, 1, 0])).as_matrix()[None, None, ...], [BATCH_SIZE, IMAGE_NUM, 1, 1]), src_images.dtype)
    theoretical, numerical = tf.test.compute_gradient(test_check, [src_images, base_plane, offsets, Rs, Ts])
    np.testing.assert_allclose(theoretical[0] , numerical[0], rtol=5e-5, atol=1e-6)
    np.testing.assert_allclose(theoretical[1] , numerical[1], rtol=5e-5, atol=1e-6)
    # np.testing.assert_allclose(theoretical[2] , numerical[2], rtol=5e-5, atol=1e-6)
    print("4: ",np.max(theoretical[0]), np.max(theoretical[1]), np.max(theoretical[2]))

if __name__ == '__main__':
  test.main()
