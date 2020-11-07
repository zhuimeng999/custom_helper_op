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
from custom_helper_op import decode_pfm
import tensorflow as tf

def load_pfm(file):
    color = None
    width = None
    height = None
    scale = None
    data_type = None
    header = file.readline().decode('UTF-8').rstrip()
    import re
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('UTF-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    # scale = float(file.readline().rstrip())
    scale = float((file.readline()).decode('UTF-8').rstrip())
    if scale < 0:  # little-endian
        data_type = '<f'
    else:
        data_type = '>f'  # big-endian
    data_string = file.read()
    data = np.fromstring(data_string, data_type)
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = cv2.flip(data, 0)
    return data

class DecoderTest(test_util.parameterized.TestCase):
  @test_util.parameterized.parameters(
    {'filename': '/home/lucius/data/mvs_training/dtu/Depths/scan1_train/depth_map_0000.pfm'},
    {'filename': '/home/lucius/data/mvs_training/dtu/Depths/scan1_train/depth_map_0001.pfm'}
  )
  def testOfnDecoder(self, filename):
    test_data = cv2.imread(filename, cv2.IMREAD_UNCHANGED)[:, :, None]
    image = tf.io.read_file(filename)
    # output = load_pfm(open(filename, 'rb'))[:, :, None]
    output = decode_pfm(image).numpy()
    # output = np.transpose(output.reshape((160, 128, 1)), [1, 0, 2])
    # cv2.imshow('a', test_data)
    cv2.imwrite('/tmp/output.pfm', output)
    # cv2.waitKey(0)
    np.testing.assert_almost_equal(test_data, output)

if __name__ == '__main__':
  test.main()
