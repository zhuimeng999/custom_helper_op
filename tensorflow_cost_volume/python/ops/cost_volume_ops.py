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
from tensorflow_cost_volume.python.utils.path_helper import package_root
import tensorflow as tf

cost_volume_ops = load_library.load_op_library(
    os.path.join(package_root, '_cost_volume.so'))

@tf.function
def cost_volume(images, transforms, name=None):
    with tf.name_scope(name or "cost_volume"):
        images_tensor = tf.convert_to_tensor(images, name="data")
        transforms_tensor = tf.convert_to_tensor(transforms, name="warp")
        return cost_volume_ops.cost_volume(images=images_tensor, transforms=transforms_tensor, interpolation='BILINEAR')


@tf.RegisterGradient("CostVolume")
def _cost_volume_grad(op, grad_output):
    images_tensor, transforms_tensor = op.inputs
    grad_output_tensor = tf.convert_to_tensor(grad_output, name="grad_output")
    image_grad = cost_volume_ops.cost_volume_grad(images=images_tensor, transforms=transforms_tensor, grad=grad_output_tensor, interpolation='BILINEAR')
    return [image_grad, None]


tf.no_gradient("CostVolumeGrad")
