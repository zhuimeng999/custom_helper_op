#!/usr/bin/python3
# -*- coding: UTF-8 -*-

import tensorflow as tf
from custom_helper_op.python.ops.custom_helper_ops import index_initializer
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
         