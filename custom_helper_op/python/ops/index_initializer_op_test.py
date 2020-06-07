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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
from tensorflow.python.platform import test
from tensorflow.python.framework import test_util
from custom_helper_op import index_initializer
import tensorflow as tf

class IndexInitializerTest(test_util.parameterized.TestCase):
  @test_util.parameterized.parameters(
    {'out_shape': [5, 6]},
    {'out_shape': [7, 8]},
    {'out_shape': [1, 1]},
    {'out_shape': [1, 8]},
    {'out_shape': [8, 1]},
    {'out_shape': [1000, 1000]}
  )
  def testIndexInitializer(self, out_shape = [3, 4]):
    test_data = np.empty((out_shape[0], out_shape[1], 3), dtype=np.float)
    for i in range(out_shape[0]):
        for j in range(out_shape[1]):
            test_data[i, j, 0] = i
            test_data[i, j, 1] = j
            test_data[i, j, 2] = 1


    with tf.device('/cpu:0'):
      output = tf.Variable(index_initializer()(output_shape=out_shape , dtype=tf.float32))
    np.testing.assert_almost_equal(test_data, output.numpy())

    with tf.device('/gpu:0'):
      output = tf.Variable(index_initializer()(output_shape=out_shape , dtype=tf.float32))
    np.testing.assert_almost_equal(test_data, output.numpy())
    #self.assertAlmostEqual(t, delta=1e-8)

if __name__ == '__main__':
  test.main()
