import tensorflow as tf
from tensorflow.python.platform import test
from absl.testing import parameterized
from custom_helper_op import sparse_conv2d, sparse_conv3d, SparseConv3DLayer, sparse_pad
import numpy as np
from tensorflow.python.ops import gradient_checker_v2
import time


class SparseConv3DTest(test.TestCase, parameterized.TestCase):
    @parameterized.parameters(
      (1, 14, 18, 8, 5, 5, (3, 3, 3), (2, 2, 2)),
      (2, 12, 16, 20, 3, 3, (3, 3, 3), (1, 1, 1)),
      (3, 26, 32, 12, 2, 2, (3, 3, 3), (2, 2, 2)),
      (1, 30, 27, 6, 6, 6, (3, 3, 3), (3, 3, 2)),
      (2, 10, 18, 9, 8, 8, (3, 3, 3), (2, 2, 3)),
      (3, 30, 20, 15, 4, 4, (3, 3, 3), (5, 4, 3)),
    )
    def testForward(self, BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDES_SIZE):
        images = tf.random.uniform([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, IN_CHANNELS], dtype=tf.float32)
        filters = np.zeros([KERNEL_SIZE[0], KERNEL_SIZE[1], KERNEL_SIZE[2], IN_CHANNELS, OUT_CHANNELS], dtype=np.float32)
        for c in range(OUT_CHANNELS):
          filters[1, 1, 1, c, c] = 1.
        filters = tf.constant(filters)
        base_plane = tf.random.uniform([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 1], minval=0, maxval=10, dtype=tf.int32)
        # default_value = tf.random.uniform([], dtype=images.dtype)
        default_value = tf.constant(0.)
        out = sparse_conv3d(images, filters, default_value, base_plane, strides=STRIDES_SIZE, dilations=(1, 1, 1))
        out = sparse_pad(out, base_plane, strides=STRIDES_SIZE, dilations=(1, 1, 1))

        z = tf.zeros((OUT_CHANNELS,))
        for b in range(BATCH_SIZE):
          for h in range(IMAGE_HEIGHT):
            for w in range(IMAGE_WIDTH):
              for d in range(IMAGE_DEPTH):
                if ((h % STRIDES_SIZE[0]) == 0) and ((w % STRIDES_SIZE[1]) == 0) and (((d + base_plane[b, h, w, 0]) % STRIDES_SIZE[2]) == 0):
                  self.assertAllClose(out[b, h, w, d], images[b, h, w, d])
                else:
                  self.assertAllClose(out[b, h, w, d], z)


    @parameterized.parameters(
      (2, 14, 18, 8, 1, 1, (3, 3, 3), (2, 2, 2)),
      (1, 12, 16, 20, 3, 3, (3, 3, 3), (1, 1, 1)),
      (1, 26, 32, 12, 2, 2, (3, 3, 3), (2, 2, 2)),
      (1, 30, 27, 6, 6, 6, (3, 3, 3), (3, 3, 2)),
      (1, 10, 18, 9, 8, 8, (3, 3, 3), (2, 2, 3)),
      (1, 30, 20, 15, 4, 4, (3, 3, 3), (5, 4, 3)),
    )
    def testGradientFloat64(self, BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, STRIDES_SIZE):
        images = tf.random.uniform([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, IN_CHANNELS], dtype=tf.float32)
        filters = np.zeros([KERNEL_SIZE[0], KERNEL_SIZE[1], KERNEL_SIZE[2], IN_CHANNELS, OUT_CHANNELS], dtype=np.float32)
        for c in range(OUT_CHANNELS):
          filters[1, 1, 1, c, c] = 1.
        filters = tf.constant(filters)
        base_plane = tf.random.uniform([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 1], minval=0, maxval=10, dtype=tf.int32)
        # default_value = tf.random.uniform([], dtype=images.dtype)
        default_value = tf.constant(0., dtype=images.dtype)
        images = sparse_conv3d(images, filters, default_value, base_plane, strides=STRIDES_SIZE, dilations=(1, 1, 1))
        images = tf.cast(images, tf.float64)
        
        @tf.function
        def test_check(*args):
            out = sparse_pad(*args, base_plane, strides=STRIDES_SIZE, dilations=(1, 1, 1))
            return tf.reduce_mean(out)

        theoretical, numerical = gradient_checker_v2.compute_gradient(test_check, [images,])

        # with self.cached_session():
        #     # res = sparse_conv2d(images, filters, base_plane, default_value, offsets)
        #     theoretical, numerical = gradient_checker_v2.compute_gradient(test_check, [images,])
        #     # err = gradient_checker_v2.max_error(theoretical, numerical)
        self.assertAllClose(theoretical[0], numerical[0])
        # # self.assertAllClose(theoretical[3], numerical[3])

if __name__ == "__main__":
  test.main()