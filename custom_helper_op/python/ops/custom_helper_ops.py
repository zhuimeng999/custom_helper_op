#!/usr/bin/python3
# -*- coding: UTF-8 -*-

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
"""Use time_two ops in python."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
from custom_helper_op.python.utils.path_helper import package_root
import tensorflow as tf

_custom_helper_ops = load_library.load_op_library(
    os.path.join(package_root, '_custom_helper_ops.so'))

@tf.function
def cost_volume(images, transforms, name=None):
    with tf.name_scope(name or "cost_volume"):
        images_tensor = tf.convert_to_tensor(images, name="data")
        transforms_tensor = tf.convert_to_tensor(transforms, name="warp")
        return _custom_helper_ops.cost_volume(images=images_tensor, transforms=transforms_tensor, interpolation='BILINEAR')


@tf.RegisterGradient("CostVolume")
def _cost_volume_grad(op, grad_out, grad_mask):
    images_tensor, transforms_tensor = op.inputs
    _, transformed_mask = op.outputs
    grad_output_tensor = tf.convert_to_tensor(grad_out, name="grad_output")
    image_grad = _custom_helper_ops.cost_volume_grad(images=images_tensor, transforms=transforms_tensor, transformed_mask=transformed_mask, grad=grad_output_tensor, interpolation='BILINEAR')
    return [image_grad, None]


def decode_pfm(contents, name=None):
    """
    Decode a PNM-encoded image to a uint8 tensor.
    Args:
      contents: A `Tensor` of type `string`. 0-D.  The PNM-encoded image.
      name: A name for the operation (optional).
    Returns:
      A `Tensor` of type `uint8` and shape of `[height, width, 4]` (RGBA).
    """
    return _custom_helper_ops.decode_pfm(contents, name=name)


tf.no_gradient("CostVolumeGrad")

class index_initializer(tf.initializers.Initializer):
  def __call__(self, output_shape, dtype=tf.float32):
    """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only numeric or boolean dtypes are
       supported.

    Raises:
      ValuesError: If the dtype is not numeric or boolean.
    """
    return _custom_helper_ops.index_initializer(output_shape=output_shape, dtype=dtype)
