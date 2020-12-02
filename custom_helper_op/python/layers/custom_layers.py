#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import tensorflow as tf
from custom_helper_op.python.ops.custom_helper_ops import index_initializer, cost_aggregate, sparse_conv3d, cost_volume, cost_volume_v2
from tensorflow_addons.image import resampler

class DepthProjectLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(DepthProjectLayer, self).__init__(**kwargs)


    def build(self, input_shape):
        image_shape, _, _ = input_shape
        image_shape.assert_has_rank(4)
        self.base_index = self.add_weight(shape=(image_shape[1], image_shape[2], 3),
                             initializer=index_initializer,
                             trainable=False)

    def call(self, inputs):
        image_tensor, depth_tensor, project_tensor = inputs
        rotate_tensor = tf.transpose(tf.tensordot(project_tensor[:, :3, :3], self.base_index, axes=[-1, -1]), [0, 2, 3, 1])
        sample_index =  rotate_tensor*depth_tensor[:, :, :, None] + project_tensor[:, None, None, :3, 3]
        sample_index =  tf.divide(sample_index[:, :, :, :2], sample_index[:, :, :, 2:3])
        return resampler(image_tensor, sample_index)

class CostMapLayer(tf.keras.layers.Layer):
    def __init__(self, reduce_method="MIN", half_centor=True, default_type='DYNAMIC', **kwargs):
        super(CostMapLayer, self).__init__(**kwargs)
        assert reduce_method in ["MEAN", "MIN"]

        self.reduce_method = reduce_method
        self.half_centor = half_centor
        self.default_type = default_type


    def build(self, input_shape):
        if self.default_type == 'DYNAMIC':
            self.default_cost = self.add_weight(shape=[], initializer="zeros", trainable=True, name='CostMapDefault')
        elif self.default_type == 'CONSTANT':
            self.default_cost = self.add_weight(shape=[], initializer="zeros", trainable=False, name='CostMapDefault')
        else:
            self.default_value = self.default_type

    def call(self, inputs, **kwargs):
        cost, cost_mask = cost_aggregate(*inputs, reduce_method=self.reduce_method, half_centor=self.half_centor)
        if self.reduce_method == "MEAN":
            cost = tf.where(cost_mask > 0, cost, self.default_cost)
        else:
            cost = tf.where(cost_mask >= 0, cost, self.default_cost)
        return cost, cost_mask

class CostMapLayerV2(tf.keras.layers.Layer):
    def __init__(self, reduce_method="MIN", half_centor=True, default_type='DYNAMIC', groups=1, **kwargs):
        super(CostMapLayerV2, self).__init__(**kwargs)
        assert reduce_method in ["MEAN", "MIN"]

        self.reduce_method = reduce_method
        self.half_centor = half_centor
        self.default_type = default_type
        self.groups = groups

    def build(self, input_shape):
        if self.default_type == 'DYNAMIC':
            self.default_cost = self.add_weight(shape=[], initializer="zeros", trainable=True, name='CostMapDefault')
        elif self.default_type == 'CONSTANT':
            self.default_cost = self.add_weight(shape=[], initializer="zeros", trainable=False, name='CostMapDefault')
        else:
            self.default_value = self.default_type

    def call(self, inputs, **kwargs):
        cost, cost_mask = cost_volume_v2(*inputs, reduce_method=self.reduce_method, groups=self.groups, half_centor=self.half_centor)
        if self.reduce_method == "MEAN":
            cost = tf.where(cost_mask > 0, cost, self.default_cost)
        else:
            cost = tf.where(cost_mask >= 0, cost, self.default_cost)
        return cost, cost_mask

class SparseConv3DLayer(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, default_type='CONSTANT', strides=[1, 1, 1], dilations=[1, 1, 1], use_bias=False, **kwargs):
        super(SparseConv3DLayer, self).__init__(**kwargs)
        assert len(kernel_size) == 3
        assert len(strides) == 3
        assert len(dilations) == 3

        assert strides[0] == 1
        assert strides[1] == 1
        assert strides[2] == 1

        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.dilations = dilations
        self.use_bias = use_bias
        self.default_type = default_type


    def build(self, input_shape):
        images, _ = input_shape
        self.kernel = self.add_weight(shape=(*self.kernel_size, images[-1], self.filters),
                             initializer='glorot_uniform',
                             trainable=False,
                             dtype=self.dtype,
                             name='sparse3d_kernal')
        if self.use_bias:
            self.bias = self.add_weight(
                name='sparse3d_bias',
                shape=(self.filters,),
                initializer='zeros',
                dtype=self.dtype,
                trainable=True)
        if self.default_type == 'CONSTANT':
            self.default_value = self.add_weight(shape=[], initializer='zeros', trainable=False, dtype=self.dtype, name='sparse_conv3d_default')
        elif self.default_type == 'DYNAMIC':
            self.default_value = self.add_weight(shape=[], initializer='zeros', trainable=True, dtype=self.dtype, name='sparse_conv3d_default')
        else:
            self.default_value = self.default_type

    def call(self, inputs, **kwargs):
        images, base_plane = inputs
        out = sparse_conv3d(images, self.kernel, self.default_value, base_plane, strides=self.strides, dilations=self.dilations)
        if self.use_bias:
            out =  tf.nn.bias_add(out, self.bias)
        return out