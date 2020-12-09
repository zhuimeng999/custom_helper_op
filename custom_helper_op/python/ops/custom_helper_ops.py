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
def cost_volume(ref_image, src_images, base_plane, offsets, Rs, Ts, reduce_method="MEAN", half_centor=True, name=None):
    with tf.name_scope(name or "cost_aggregate"):
        offsets = tf.convert_to_tensor(offsets, name="offsets")
        rs = tf.convert_to_tensor(Rs, name="Rs")
        ts = tf.convert_to_tensor(Ts, name="Ts")
        return _custom_helper_ops.cost_volume(ref_image=ref_image, src_images=src_images, base_plane=base_plane, offsets=offsets, rs=rs, ts=ts, reduce_method=reduce_method, half_centor=half_centor)


@tf.RegisterGradient("CostVolume")
def _cost_volume_grad(op, grad_out, grad_mask):
    ref_image, src_images, base_plane, offsets, Rs, Ts = op.inputs
    cost, cost_mask = op.outputs

    offsets = tf.convert_to_tensor(offsets, name="offsets")
    rs = tf.convert_to_tensor(Rs, name="Rs")
    ts = tf.convert_to_tensor(Ts, name="Ts")

    ref_image_grad, src_images_grad, base_plane_grad = _custom_helper_ops.cost_volume_grad(ref_image=ref_image, 
                                                                                          src_images=src_images, 
                                                                                          base_plane=base_plane, 
                                                                                          offsets=offsets, 
                                                                                          rs=rs, 
                                                                                          ts=ts,
                                                                                          cost_grad=grad_out,
                                                                                          cost_mask=cost_mask,
                                                                                          reduce_method=op.get_attr("reduce_method"),
                                                                                          half_centor=op.get_attr("half_centor")
                                                                                          )
    return [ref_image_grad, src_images_grad, base_plane_grad, None, None, None]

tf.no_gradient("CostVolumeGrad")

@tf.function
def cost_volume_v2(ref_image, src_images, depth_grid, Rs, Ts, reduce_method="MEAN", groups=None, half_centor=True, name=None):
    with tf.name_scope(name or "cost_volume_v2"):
        depth_grid = tf.convert_to_tensor(depth_grid, name="depth_grid")
        rs = tf.convert_to_tensor(Rs, name="Rs")
        ts = tf.convert_to_tensor(Ts, name="Ts")
        return _custom_helper_ops.cost_volume_v2(ref_image=ref_image, src_images=src_images, depth_grid=depth_grid, rs=rs, ts=ts, reduce_method=reduce_method, groups=groups, half_centor=half_centor)


@tf.RegisterGradient("CostVolumeV2")
def _cost_volume_grad_v2(op, grad_out, grad_mask):
    ref_image, src_images, depth_grid, Rs, Ts = op.inputs
    cost, cost_mask = op.outputs

    depth_grid = tf.convert_to_tensor(depth_grid, name="depth_grid")
    rs = tf.convert_to_tensor(Rs, name="Rs")
    ts = tf.convert_to_tensor(Ts, name="Ts")

    ref_image_grad, src_images_grad, depth_grid_grad = _custom_helper_ops.cost_volume_grad_v2(ref_image=ref_image, 
                                                                                          src_images=src_images, 
                                                                                          depth_grid=depth_grid, 
                                                                                          rs=rs, 
                                                                                          ts=ts,
                                                                                          cost_grad=grad_out,
                                                                                          cost_mask=cost_mask,
                                                                                          reduce_method=op.get_attr("reduce_method"),
                                                                                          groups=op.get_attr('groups'),
                                                                                          half_centor=op.get_attr("half_centor")
                                                                                          )
    return [ref_image_grad, src_images_grad, depth_grid_grad, None, None]

tf.no_gradient("CostVolumeGradV2")

@tf.function
def cost_volume_v3(ref_image, src_images, base_plane, offsets, Rs, Ts, reduce_method="MEAN", groups=None, half_centor=True, name=None):
    with tf.name_scope(name or "cost_volume_v3"):
        offsets = tf.convert_to_tensor(offsets, name="offsets")
        rs = tf.convert_to_tensor(Rs, name="Rs")
        ts = tf.convert_to_tensor(Ts, name="Ts")
        return _custom_helper_ops.cost_volume_v3(ref_image=ref_image, src_images=src_images, base_plane=base_plane, offsets=offsets, rs=rs, ts=ts, reduce_method=reduce_method, groups=groups, half_centor=half_centor)


@tf.RegisterGradient("CostVolumeV3")
def _cost_volume_grad_v3(op, grad_out, grad_mask):
    ref_image, src_images, base_plane, offsets, Rs, Ts = op.inputs
    cost, cost_mask = op.outputs

    offsets = tf.convert_to_tensor(offsets, name="offsets")
    rs = tf.convert_to_tensor(Rs, name="Rs")
    ts = tf.convert_to_tensor(Ts, name="Ts")

    ref_image_grad, src_images_grad, base_plane_grad = _custom_helper_ops.cost_volume_grad_v3(ref_image=ref_image, 
                                                                                          src_images=src_images, 
                                                                                          base_plane=base_plane, 
                                                                                          offsets=offsets, 
                                                                                          rs=rs, 
                                                                                          ts=ts,
                                                                                          cost_grad=grad_out,
                                                                                          cost_mask=cost_mask,
                                                                                          reduce_method=op.get_attr("reduce_method"),
                                                                                          groups=op.get_attr('groups'),
                                                                                          half_centor=op.get_attr("half_centor")
                                                                                          )
    return [ref_image_grad, src_images_grad, base_plane_grad, None, None, None]

tf.no_gradient("CostVolumeGradV3")

@tf.function
def cost_aggregate(ref_image, src_images, base_plane, offsets, Rs, Ts, reduce_method="MEAN", half_centor=True, name=None):
    with tf.name_scope(name or "cost_aggregate"):
        offsets = tf.convert_to_tensor(offsets, name="offsets")
        rs = tf.convert_to_tensor(Rs, name="Rs")
        ts = tf.convert_to_tensor(Ts, name="Ts")
        return _custom_helper_ops.cost_aggregate(ref_image=ref_image, src_images=src_images, base_plane=base_plane, offsets=offsets, rs=rs, ts=ts, reduce_method=reduce_method, half_centor=half_centor)


@tf.RegisterGradient("CostAggregate")
def _cost_aggregate_grad(op, grad_out, grad_mask):
    ref_image, src_images, base_plane, offsets, Rs, Ts = op.inputs
    cost, cost_mask = op.outputs

    offsets = tf.convert_to_tensor(offsets, name="offsets")
    rs = tf.convert_to_tensor(Rs, name="Rs")
    ts = tf.convert_to_tensor(Ts, name="Ts")

    ref_image_grad, src_images_grad, base_plane_grad = _custom_helper_ops.cost_aggregate_grad(ref_image=ref_image, 
                                                                                          src_images=src_images, 
                                                                                          base_plane=base_plane, 
                                                                                          offsets=offsets, 
                                                                                          rs=rs, 
                                                                                          ts=ts,
                                                                                          cost_grad=grad_out,
                                                                                          cost_mask=cost_mask,
                                                                                          reduce_method=op.get_attr("reduce_method"),
                                                                                          half_centor=op.get_attr("half_centor")
                                                                                          )
    return [ref_image_grad, src_images_grad, base_plane_grad, None, None, None]

tf.no_gradient("CostAggregateGrad")

@tf.function
def feature_aggregate(src_images, base_plane, offsets, Rs, Ts, half_centor=True, name=None):
    with tf.name_scope(name or "feature_aggregate"):
        offsets = tf.convert_to_tensor(offsets, name="offsets")
        rs = tf.convert_to_tensor(Rs, name="Rs")
        ts = tf.convert_to_tensor(Ts, name="Ts")
        return _custom_helper_ops.feature_aggregate(src_images=src_images, base_plane=base_plane, offsets=offsets, rs=rs, ts=ts, half_centor=half_centor)


@tf.RegisterGradient("FeatureAggregate")
def _feature_aggregate_grad(op, grad_out, grad_mask):
    src_images, base_plane, offsets, Rs, Ts = op.inputs
    mapped_feature, mapped_mask = op.outputs

    offsets = tf.convert_to_tensor(offsets, name="offsets")
    rs = tf.convert_to_tensor(Rs, name="Rs")
    ts = tf.convert_to_tensor(Ts, name="Ts")

    src_images_grad, base_plane_grad = _custom_helper_ops.feature_aggregate_grad(src_images=src_images, 
                                                                                          base_plane=base_plane, 
                                                                                          offsets=offsets, 
                                                                                          rs=rs, 
                                                                                          ts=ts,
                                                                                          mapped_feature_grad=grad_out,
                                                                                          mapped_mask=mapped_mask,
                                                                                          half_centor=op.get_attr("half_centor")
                                                                                          )
    return [src_images_grad, base_plane_grad, None, None, None]

tf.no_gradient("FeatureAggregateGrad")


@tf.function
def sparse_conv2d(images, filter, base_plane, default_value, offsets, strides=(1, 1), dilations=(1, 1), name=None):
    with tf.name_scope(name or "sparse_conv2d"):
        return _custom_helper_ops.sparse_conv2d(images=images, filter=filter, base_plane=base_plane, default_value=default_value, offsets=offsets, strides=strides, dilations=dilations)


@tf.RegisterGradient("SparseConv2D")
def _sparse_conv2d_grad(op, out_grad):
    images, filter, base_plane, default_value, offsets = op.inputs
    # cost, cost_mask = op.outputs
    images_grad, filter_grad, base_plane_grad, default_value_grad = _custom_helper_ops.sparse_conv2d_grad(images=images, 
                                                                                          filter=filter, 
                                                                                          base_plane=base_plane, 
                                                                                          default_value=default_value, 
                                                                                          offsets=offsets, 
                                                                                          out_grad=out_grad,
                                                                                          strides=op.get_attr("strides"),
                                                                                          dilations=op.get_attr("dilations")
                                                                                          )
    return [images_grad, filter_grad, base_plane_grad, default_value_grad, None]

tf.no_gradient("SparseConv2DGrad")

@tf.function
def sparse_pad(images, base_plane, strides=(1, 1, 1), dilations=(1, 1, 1), name=None):
    with tf.name_scope(name or "sparse_pad"):
        return _custom_helper_ops.sparse_pad(images=images, base_plane=base_plane, strides=strides, dilations=dilations)


@tf.RegisterGradient("SparsePad")
def _sparse_pad_grad(op, out_grad):
    images, base_plane = op.inputs
    images_grad = _custom_helper_ops.sparse_pad_grad(images=images, base_plane=base_plane, out_grad=out_grad,
                                                                                          strides=op.get_attr("strides"),
                                                                                          dilations=op.get_attr("dilations")
                                                                                          )

    return [images_grad, None]

tf.no_gradient("SparsePadGrad")

@tf.function
def sparse_conv3d(images, filters, default_value, base_plane, strides=(1, 1, 1), dilations=(1, 1, 1), dynamic_default=False, name=None):
    with tf.name_scope(name or "sparse_conv3d"):
        return _custom_helper_ops.sparse_conv3d(images=images, filters=filters, default_value=default_value, base_plane=base_plane, strides=strides, dilations=dilations, dynamic_default=dynamic_default)


@tf.RegisterGradient("SparseConv3D")
def _sparse_conv3d_grad(op, out_grad):
    images, filters, default_value, base_plane = op.inputs
    # cost, cost_mask = op.outputs
    images_grad, filter_grad, default_value_grad = _custom_helper_ops.sparse_conv3d_grad(images=images, 
                                                                                          filters=filters, 
                                                                                          default_value=default_value,
                                                                                          base_plane=base_plane, 
                                                                                          out_grad=out_grad,
                                                                                          strides=op.get_attr("strides"),
                                                                                          dilations=op.get_attr("dilations"),
                                                                                          dynamic_default=op.get_attr("dynamic_default")
                                                                                          )
    if op.get_attr("dynamic_default") is False:
        default_value_grad = None
    return [images_grad, filter_grad, default_value_grad, None]

tf.no_gradient("SparseConv3DGrad")

@tf.function
def sparse_conv3d_fast(images, filters, default_value, base_plane, data_format='NDHWC', strides=(1, 1, 1), dilations=(1, 1, 1), dynamic_default=False, name=None):
    with tf.name_scope(name or "sparse_conv3d_fast"):
        return _custom_helper_ops.sparse_conv3d_fast(images=images, filters=filters, default_value=default_value, base_plane=base_plane, 
                                                        data_format=data_format, strides=strides, dilations=dilations, dynamic_default=dynamic_default)


@tf.RegisterGradient("SparseConv3DFast")
def _sparse_conv3d_fast_grad(op, out_grad):
    images, filters, default_value, base_plane = op.inputs
    # cost, cost_mask = op.outputs
    images_grad, filter_grad, default_value_grad = _custom_helper_ops.sparse_conv3d_fast_grad(images=images, 
                                                                                          filters=filters, 
                                                                                          default_value=default_value,
                                                                                          base_plane=base_plane, 
                                                                                          out_grad=out_grad,
                                                                                          data_format=op.get_attr('data_format'),
                                                                                          strides=op.get_attr("strides"),
                                                                                          dilations=op.get_attr("dilations"),
                                                                                          dynamic_default=op.get_attr("dynamic_default")
                                                                                          )
    if op.get_attr("dynamic_default") is False:
        default_value_grad = None
    return [images_grad, filter_grad, default_value_grad, None]

tf.no_gradient("SparseConv3DFastGrad")

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


def index_initializer(output_shape, half_centor=True, dtype=tf.float32):
      return _custom_helper_ops.index_initializer(output_shape=output_shape, half_centor=half_centor, dtype=dtype)

class IndexInitializer(tf.initializers.Initializer):
  def __call__(self, output_shape, half_centor=True, dtype=tf.float32):
    """Returns a tensor object initialized as specified by the initializer.

    Args:
      shape: Shape of the tensor.
      dtype: Optional dtype of the tensor. Only numeric or boolean dtypes are
       supported.

    Raises:
      ValuesError: If the dtype is not numeric or boolean.
    """
    return _custom_helper_ops.index_initializer(output_shape=output_shape, half_centor=half_centor, dtype=dtype)
