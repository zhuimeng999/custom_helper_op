#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import tensorflow as tf
from custom_helper_op.python.ops.custom_helper_ops import index_initializer, cost_aggregate
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
    def __init__(self, reduce_method="MIN", half_centor=True, **kwargs):
        super(CostMapLayer, self).__init__(**kwargs)
        assert reduce_method in ["MEAN", "MIN"]

        self.default_cost = self.add_weight('default_cost', 1, initializer="zeros", trainable=True)
        self.reduce_method = reduce_method
        self.half_centor = half_centor

    def call(self, inputs, **kwargs):
        cost, cost_mask = cost_aggregate(*inputs, reduce_method=self.reduce_method, half_centor=self.half_centor)
        if self.reduce_method == "MEAN":
            cost = tf.where(cost_mask > 0, cost, self.default_cost)
        else:
            cost = tf.where(cost_mask >= 0, cost, self.default_cost)
        return cost, cost_mask

class SparseConv2DLayer(tf.keras.layers.Layer):
    def __init__(self, reduce_method="MIN", half_centor=True, **kwargs):
        super(SparseConv2DLayer, self).__init__(**kwargs)
        assert reduce_method in ["MEAN", "MIN"]

        self.default_cost = self.add_weight('default_cost', 1, initializer="zeros", trainable=True)
        self.reduce_method = reduce_method
        self.half_centor = half_centor

    def call(self, inputs, **kwargs):
        cost, cost_mask = cost_aggregate(*inputs, reduce_method=self.reduce_method, half_centor=self.half_centor)
        if self.reduce_method == "MEAN":
            cost = tf.where(cost_mask > 0, cost, self.default_cost)
        else:
            cost = tf.where(cost_mask >= 0, cost, self.default_cost)
        return cost, cost_mask
         