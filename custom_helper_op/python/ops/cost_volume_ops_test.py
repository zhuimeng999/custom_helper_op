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
from custom_helper_op import cost_volume
import tensorflow as tf

def getIdentifiy(shape):
  identifyMat = np.array([1., 0., 0., 0., 1., 0., 0., 0.], dtype=np.float32)
  transforms = np.empty(shape, dtype=np.float32)
  for b in range(shape[0]):
    for n in range(shape[1]):
      for d in range(shape[2]):
        transforms[b, n, d, :] = identifyMat
  return transforms

def getRotation90(shape):
  identifyMat = np.array([0., 1., 0., 1., 0., 0., 0., 0.], dtype=np.float32)
  transforms = np.empty(shape, dtype=np.float32)
  for b in range(shape[0]):
    for n in range(shape[1]):
      for d in range(shape[2]):
        transforms[b, n, d, :] = identifyMat
  return transforms

class MyOperatorTest(test_util.parameterized.TestCase):
  @test_util.parameterized.parameters(
    {'BATCH_SIZE':1, 'IMAGE_NUM':2, 'IMAGE_HEIGHT':10, 'IMAGE_WIDTH':20, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':9},
    {'BATCH_SIZE':1, 'IMAGE_NUM':3, 'IMAGE_HEIGHT':10, 'IMAGE_WIDTH':20, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':9},
    {'BATCH_SIZE':1, 'IMAGE_NUM':10, 'IMAGE_HEIGHT':10, 'IMAGE_WIDTH':20, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':9},
    {'BATCH_SIZE':2, 'IMAGE_NUM':2, 'IMAGE_HEIGHT':10, 'IMAGE_WIDTH':20, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':9},
    {'BATCH_SIZE':2, 'IMAGE_NUM':3, 'IMAGE_HEIGHT':10, 'IMAGE_WIDTH':20, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':9},
    {'BATCH_SIZE':3, 'IMAGE_NUM':10, 'IMAGE_HEIGHT':10, 'IMAGE_WIDTH':20, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':9}
  )
  def testCostVolumeIdentify(self, BATCH_SIZE = 1, IMAGE_NUM = 2, IMAGE_HEIGHT = 10, IMAGE_WIDTH = 20, IMAGE_CHANNELS = 32, IMAGE_DEPTH = 256):
    images = np.random.random([BATCH_SIZE, IMAGE_NUM, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]).astype(np.float32)
    transforms = getIdentifiy([BATCH_SIZE, IMAGE_NUM - 1, IMAGE_DEPTH, 8])
    images_tensor = tf.constant(images)
    transforms_tensor = tf.constant(transforms)
    with tf.GradientTape() as tape:
      cost, mask = cost_volume(images_tensor, transforms_tensor)

    gradients = tape.gradient(cost, images_tensor)
    self.assertEqual(cost.shape, np.array([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, IMAGE_CHANNELS]))

    cost_out = cost.numpy()
    for b in range(BATCH_SIZE):
            for h in range(1, IMAGE_HEIGHT - 1):
                for w in range(1, IMAGE_WIDTH - 1):
                    for d in range(IMAGE_DEPTH):
                        for c in range(IMAGE_CHANNELS):
                            tmp = 0.
                            for n in range(1, IMAGE_NUM):
                                diff = images[b, n, h, w, c] - images[b, 0, h, w, c]
                                tmp = tmp + diff*diff
                            tmp = tmp / (IMAGE_NUM - 1)
                            self.assertAlmostEqual(tmp, cost_out[b, h, w, d, c], msg='current index' + str([b, h, w, d, c]), delta=1e-6)

  @test_util.parameterized.parameters(
    {'BATCH_SIZE':1, 'IMAGE_NUM':2, 'IMAGE_HEIGHT':10, 'IMAGE_WIDTH':20, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':9},
    {'BATCH_SIZE':1, 'IMAGE_NUM':3, 'IMAGE_HEIGHT':10, 'IMAGE_WIDTH':20, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':9},
    {'BATCH_SIZE':1, 'IMAGE_NUM':10, 'IMAGE_HEIGHT':10, 'IMAGE_WIDTH':20, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':9},
    {'BATCH_SIZE':2, 'IMAGE_NUM':2, 'IMAGE_HEIGHT':10, 'IMAGE_WIDTH':20, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':9},
    {'BATCH_SIZE':2, 'IMAGE_NUM':3, 'IMAGE_HEIGHT':10, 'IMAGE_WIDTH':20, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':9},
    {'BATCH_SIZE':3, 'IMAGE_NUM':10, 'IMAGE_HEIGHT':10, 'IMAGE_WIDTH':20, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':9}
  )
  def testCostVolumeGradIdentify(self, BATCH_SIZE = 1, IMAGE_NUM = 2, IMAGE_HEIGHT = 10, IMAGE_WIDTH = 20, IMAGE_CHANNELS = 32, IMAGE_DEPTH = 256):
    images = np.random.random([BATCH_SIZE, IMAGE_NUM, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]).astype(np.float32)
    transforms = getIdentifiy([BATCH_SIZE, IMAGE_NUM - 1, IMAGE_DEPTH, 8])
    images_tensor = tf.constant(images)
    transforms_tensor = tf.constant(transforms)
    with tf.GradientTape() as tape:
      tape.watch(images_tensor)
      cost, mask = cost_volume(images_tensor, transforms_tensor)

    gradients = tape.gradient(cost, [images_tensor])[0]
    self.assertEqual(cost.shape, np.array([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, IMAGE_CHANNELS]))

    cost_out = cost.numpy()
    grad_out = gradients.numpy()
    image_grad = np.zeros(images.shape, dtype=np.float)
    self.assertEqual(image_grad.shape, images.shape)
    self.assertEqual(grad_out.shape, images.shape)
    for b in range(BATCH_SIZE):                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
            for h in range(1, IMAGE_HEIGHT):
                for w in range(1, IMAGE_WIDTH):
                    for d in range(IMAGE_DEPTH):
                        for c in range(IMAGE_CHANNELS):
                            tmp = 0.
                            for n in range(1, IMAGE_NUM):
                                diff = images[b, n, h, w, c] - images[b, 0, h, w, c]
                                tmp = tmp + diff*diff
                                image_grad[b, n, h, w, c] += 2*diff
                                image_grad[b, 0, h, w, c] += -2*diff
                            tmp = tmp / (IMAGE_NUM - 1)
    for b in range(BATCH_SIZE):
        for n in range(0, IMAGE_NUM):
            for h in range(2, IMAGE_HEIGHT - 2):
                for w in range(2, IMAGE_WIDTH - 2):
                        for c in range(IMAGE_CHANNELS):

                              self.assertAlmostEqual(image_grad[b, n, h, w, c]/(IMAGE_NUM - 1), grad_out[b, n, h, w, c], msg='current index' + str([b, h, w, d, c]), places=4)

  @test_util.parameterized.parameters(
    {'BATCH_SIZE':1, 'IMAGE_NUM':2, 'IMAGE_HEIGHT':10, 'IMAGE_WIDTH':10, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':9},
    {'BATCH_SIZE':1, 'IMAGE_NUM':3, 'IMAGE_HEIGHT':10, 'IMAGE_WIDTH':10, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':9},
    {'BATCH_SIZE':1, 'IMAGE_NUM':10, 'IMAGE_HEIGHT':10, 'IMAGE_WIDTH':10, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':9},
    {'BATCH_SIZE':2, 'IMAGE_NUM':2, 'IMAGE_HEIGHT':10, 'IMAGE_WIDTH':10, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':9},
    {'BATCH_SIZE':2, 'IMAGE_NUM':3, 'IMAGE_HEIGHT':10, 'IMAGE_WIDTH':10, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':9},
    {'BATCH_SIZE':3, 'IMAGE_NUM':10, 'IMAGE_HEIGHT':10, 'IMAGE_WIDTH':10, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':9}
  )
  def testCostVolumeRotate90(self, BATCH_SIZE = 1, IMAGE_NUM = 2, IMAGE_HEIGHT = 10, IMAGE_WIDTH = 20, IMAGE_CHANNELS = 32, IMAGE_DEPTH = 256):
    images = np.random.random([BATCH_SIZE, IMAGE_NUM, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]).astype(np.float32)
    transforms = getRotation90([BATCH_SIZE, IMAGE_NUM - 1, IMAGE_DEPTH, 8])

    images_tensor = tf.constant(images)
    transforms_tensor = tf.constant(transforms)
    with tf.GradientTape() as tape:
      tape.watch(images_tensor)
      cost, mask = cost_volume(images_tensor, transforms_tensor)

    gradients = tape.gradient(cost, images_tensor)
    # cost = cost_volume(images, transforms)
    self.assertEqual(cost.shape, np.array([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, IMAGE_CHANNELS]))

    cost_out = cost.numpy()
    for b in range(BATCH_SIZE):
            for h in range(1, IMAGE_HEIGHT - 1):
                for w in range(1, IMAGE_WIDTH - 1):
                    for d in range(IMAGE_DEPTH):
                        for c in range(IMAGE_CHANNELS):
                            tmp = 0.
                            for n in range(1, IMAGE_NUM):
                                diff = images[b, n, w, h, c] - images[b, 0, h, w, c]
                                tmp = tmp + diff*diff
                            tmp = tmp / (IMAGE_NUM - 1)
                            self.assertAlmostEqual(tmp, cost_out[b, h, w, d, c], msg='current index' + str([b, h, w, d, c]), delta=1e-6)


  @test_util.parameterized.parameters(
    {'BATCH_SIZE':1, 'IMAGE_NUM':2, 'IMAGE_HEIGHT':10, 'IMAGE_WIDTH':10, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':9},
    {'BATCH_SIZE':1, 'IMAGE_NUM':3, 'IMAGE_HEIGHT':10, 'IMAGE_WIDTH':10, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':9},
    {'BATCH_SIZE':1, 'IMAGE_NUM':10, 'IMAGE_HEIGHT':10, 'IMAGE_WIDTH':10, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':9},
    {'BATCH_SIZE':2, 'IMAGE_NUM':2, 'IMAGE_HEIGHT':10, 'IMAGE_WIDTH':10, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':9},
    {'BATCH_SIZE':2, 'IMAGE_NUM':3, 'IMAGE_HEIGHT':10, 'IMAGE_WIDTH':10, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':9},
    {'BATCH_SIZE':3, 'IMAGE_NUM':10, 'IMAGE_HEIGHT':10, 'IMAGE_WIDTH':10, 'IMAGE_CHANNELS':30, 'IMAGE_DEPTH':9}
  )
  def testCostVolumeRotateMixed(self, BATCH_SIZE = 1, IMAGE_NUM = 2, IMAGE_HEIGHT = 10, IMAGE_WIDTH = 20, IMAGE_CHANNELS = 32, IMAGE_DEPTH = 256):
    images = np.random.random([BATCH_SIZE, IMAGE_NUM, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS]).astype(np.float32)
    transforms = getRotation90([BATCH_SIZE, IMAGE_NUM - 1, IMAGE_DEPTH, 8])

    images_tensor = tf.constant(images)
    transforms_tensor = tf.constant(transforms)
    with tf.GradientTape() as tape:
      tape.watch(images_tensor)
      cost, mask = cost_volume(images_tensor, transforms_tensor)

    gradients = tape.gradient(cost, images_tensor)
    # cost = cost_volume(images, transforms)
    self.assertEqual(cost.shape, np.array([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, IMAGE_CHANNELS]))

    cost_out = cost.numpy()
    for b in range(BATCH_SIZE):
            for h in range(1, IMAGE_HEIGHT - 1):
                for w in range(1, IMAGE_WIDTH - 1):
                    for d in range(IMAGE_DEPTH):
                        for c in range(IMAGE_CHANNELS):
                            tmp = 0.
                            for n in range(1, IMAGE_NUM):
                                diff = images[b, n, w, h, c] - images[b, 0, h, w, c]
                                tmp = tmp + diff*diff
                            tmp = tmp / (IMAGE_NUM - 1)
                            self.assertAlmostEqual(tmp, cost_out[b, h, w, d, c], msg='current index' + str([b, h, w, d, c]), delta=1e-6)

if __name__ == '__main__':
  test.main()
