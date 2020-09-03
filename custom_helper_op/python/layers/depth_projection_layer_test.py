#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import numpy as np
import cv2
from tensorflow.python.platform import test
from tensorflow.python.framework import test_util
from  custom_helper_op import DepthProjectLayer
import tensorflow as tf
from scipy import interpolate

class DepthProjectTest(test_util.parameterized.TestCase):
  @test_util.parameterized.parameters(
    {'batch_size': 3, 'out_shape': [5, 6]},
    {'batch_size': 4, 'out_shape': [7, 8]},
    {'batch_size': 1, 'out_shape': [7, 8]},
    {'batch_size': 3, 'out_shape': [100, 200]},
  )
  def testIndexInitializer(self, batch_size=5, out_shape = [3, 4], out_channel=5):
    dpl = DepthProjectLayer()
    input_feature = np.random.random([batch_size, out_shape[0], out_shape[1], out_channel]).astype(np.float32)
    input_depth = np.random.random([batch_size, out_shape[0], out_shape[1]]).astype(np.float32)
    input_project = np.random.random([batch_size, 3, 4]).astype(np.float32)

    true_index = np.empty([batch_size, out_shape[0], out_shape[1], 2], dtype=np.float32)
    for i in range(batch_size):
      for h in range(out_shape[0]):
        for w in range(out_shape[1]):
          ref_pix = np.array([w, h, 1])*input_depth[i, h, w]
          ref_pix = np.array([ref_pix[0], ref_pix[1], ref_pix[2], 1])
          ref_pix = input_project[i].dot(ref_pix)
          true_index[i, h, w, 0] = ref_pix[0]/ref_pix[2]
          true_index[i, h, w, 1] = ref_pix[1]/ref_pix[2]

    true_feature = np.empty([batch_size, out_shape[0], out_shape[1], out_channel], dtype=np.float32)
    for i in range(batch_size):
      for c in range(out_channel):
        f = interpolate.interp2d(list(range(out_shape[1])), list(range(out_shape[0])), input_feature[i, :, :, c], fill_value=0)
        for h in range(out_shape[0]):
          for w in range(out_shape[1]):
            true_feature[i, h, w, c] =f(true_index[i, h, w, 0], true_index[i, h, w, 1])

    predict_feature = dpl((input_feature, input_depth, input_project))
    np.testing.assert_allclose(true_feature[:,1:-1, 1:-1, :], predict_feature.numpy()[:,1:-1, 1:-1, :], rtol=1e-5)

if __name__ == '__main__':
  test.main()
