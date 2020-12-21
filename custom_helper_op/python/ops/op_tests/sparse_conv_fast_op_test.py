import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices --tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'
# os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
# os.environ['OMP_NUM_THREADS'] = '1'
os.environ['TF_DISABLE_MKL'] = '1'

import tensorflow as tf
gpus = tf.config.experimental.get_visible_devices('GPU')

for x in gpus:
    tf.config.experimental.set_memory_growth(x, True)
# tf.config.experimental.enable_mlir_graph_optimization()

from tensorflow.python.platform import test
from absl.testing import parameterized
from custom_helper_op import sparse_conv3d_fast, sparse_conv3d_transpose_fast
import numpy as np
from tensorflow.python.ops import gradient_checker_v2
import time


class SparseConv3DFastTest(test.TestCase, parameterized.TestCase):
    # @parameterized.parameters(
    #   (1, 6, 1, 1, 1, 1, (1, 1, 1), (1, 1, 1)),
    #   (1, 33, 68, 8, 32, 4, (1, 1, 1), (1, 1, 1)),
    #   (2, 64, 80, 16, 32, 32, (2, 3, 8), (1, 1, 1)),
    #   (2, 64, 80, 16, 32, 32, (3, 3, 3), (1, 1, 1)),
    # )
    # def testForward(self, BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, DILATIONS_SIZE):
    #     depth_factor = 2
    #     out_depth = IMAGE_DEPTH*depth_factor
    #     # tf.random.set_seed(np.random.randint(0, tf.int64.max))
    #     test_strides = (2, 2, 2)

    #     in_shape = np.array((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
    #     out_shape = (in_shape + np.array(test_strides) - 1)//np.array(test_strides)

    #     full_in_shape = np.array((IMAGE_HEIGHT, IMAGE_WIDTH, out_depth))
    #     full_out_shape = (full_in_shape + np.array(test_strides) - 1)//np.array(test_strides)

    #     images_all = tf.random.uniform([BATCH_SIZE, *full_in_shape, IN_CHANNELS], dtype=tf.float32)
    #     filters = tf.random.uniform([*KERNEL_SIZE, IN_CHANNELS, OUT_CHANNELS], dtype=images_all.dtype)
    #     base_plane = tf.random.uniform([BATCH_SIZE, in_shape[0], in_shape[1], 1], minval=0, maxval=(full_in_shape[2] - in_shape[2] + 1), dtype=tf.int32)
    #     default_value = tf.random.uniform([], dtype=images_all.dtype)

    #     gather_indice = base_plane + np.arange(0, IMAGE_DEPTH, dtype=np.int32)[None, None, None, :]
    #     images = tf.gather_nd(images_all, gather_indice[..., None], batch_dims=3)
    #     mask = tf.one_hot(gather_indice, out_depth, on_value=True, off_value=False, dtype=tf.bool)
    #     mask = tf.reduce_any(mask, axis=-2)
    #     images_all = tf.where(mask[..., None], images_all, default_value)

    #     start = time.time()
    #     res = sparse_conv3d_fast(images, filters, default_value, base_plane, dilations=DILATIONS_SIZE, strides=test_strides)
    #     my_time = time.time() - start


    #     partial = (full_out_shape - 1) * test_strides + 1 - full_in_shape
    #     pad_left = np.multiply(np.array(KERNEL_SIZE)//2,np.array(DILATIONS_SIZE))
    #     pad_right = np.maximum((np.array(KERNEL_SIZE) - 1)*np.array(DILATIONS_SIZE) - pad_left + partial, np.array([0, 0, 0]))
    #     pad_size = np.stack([pad_left, pad_right], axis=-1)
    #     pad_size = np.concatenate([np.zeros((1, 2)), pad_size, np.zeros((1, 2))], axis=0)
    #     # images_nn = tf.pad(images_all, tf.constant(pad_size, dtype=tf.int32), mode="CONSTANT", constant_values=default_value)
    #     images_nn = images_all
    #     # print(pad_left, pad_right, pad_size)
    #     start = time.time()
    #     res_nn = tf.nn.conv3d(images_nn, filters, strides=(1, *test_strides, 1), padding="VALID", dilations=(1, *DILATIONS_SIZE, 1))
    #     nn_time = time.time() - start

    #     strided_base_plane = base_plane[:, 0::test_strides[0], 0::test_strides[1], :]
    #     strided_base_plane = (strided_base_plane + test_strides[2] - 1)//test_strides[2]
    #     gather_indice = strided_base_plane + np.arange(0, (IMAGE_DEPTH + test_strides[2] - 1)//test_strides[2], dtype=np.int32)[None, None, None, :]
    #     res_nn = tf.gather_nd(res_nn, gather_indice[..., None], batch_dims=3)
    #     # print(tf.shape(res), tf.shape(res_nn), tf.shape(gather_indice))
    #     # print(res, res_nn)
    #     # print(images_all, base_plane, filters)
    #     # print("my ", my_time/1000, " nn ", nn_time/1000)
    #     # test_out = tf.reduce_sum(images[:, None, :KERNEL_SIZE[0], :KERNEL_SIZE[1], :KERNEL_SIZE[2], :]*filters[None, ...], axis=(2, 3, 4, 5))
    #     # print(test_out, base_plane[:, :KERNEL_SIZE[0], :KERNEL_SIZE[1]])
    #     # print(res[:, half_kernel[0], half_kernel[1],  half_kernel[2], :])
    #     # print(res_nn[:, half_kernel[0], half_kernel[1], half_kernel[2], :])
    #     # print(res_nn, res, images_all, (base_plane[:, ::2, ::2, :] + 1)//2 )
    #     self.assertShapeEqual(res.numpy(), res_nn)
    #     self.assertAllClose(res, res_nn, rtol=1e-5)
      
    # @parameterized.parameters(
    #   (1, 6, 7, 1, 1, 1, 1, (2, 2, 1), (1, 1, 1)),
    #   (1, 8, 13, 6, 15, 3, 4, (3, 3, 3), (1, 1, 1)),
    #   (2, 4, 6, 5, 8, 2, 1, (5, 3, 3), (2, 2, 2)),
    #   (3, 4, 6, 4, 8, 2, 3, (2, 2, 1), (2, 2, 2)),
    # )
    # def testGradientFloat64(self, BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, VIRTUAL_DEPTH, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, DILATIONS_SIZE):
    #     test_strides = (1, 1, 1)

    #     tf.random.set_seed(np.random.randint(0, tf.int64.max))
    #     images_all = tf.random.uniform([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, VIRTUAL_DEPTH, IN_CHANNELS], dtype=tf.float32)
    #     # images_all = tf.ones([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, VIRTUAL_DEPTH, IN_CHANNELS], dtype=tf.float64)
    #     # images_all = np.zeros([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, VIRTUAL_DEPTH, IN_CHANNELS], dtype=np.float64)
    #     # images_all[0, 5, 5, 0, 2] = 1.
    #     # images_all = tf.constant(images_all)
    #     filters = tf.random.uniform([KERNEL_SIZE[0], KERNEL_SIZE[1], KERNEL_SIZE[2], IN_CHANNELS, OUT_CHANNELS], dtype=images_all.dtype)
    #     base_plane = tf.random.uniform([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 1], minval=0, maxval=(VIRTUAL_DEPTH - IMAGE_DEPTH + 1), dtype=tf.int32)
    #     # test_start_d = 0
    #     # base_plane = tf.ones([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 1], dtype=tf.int32) * test_start_d
    #     default_value = tf.random.uniform([], dtype=images_all.dtype)
    #     # default_value = tf.constant(0, dtype=images_all.dtype)

    #     half_kernel = np.array(KERNEL_SIZE)//2
    #     gather_indice = base_plane + np.arange(0, IMAGE_DEPTH, dtype=np.int32)[None, None, None, :]
    #     images = tf.gather_nd(images_all, gather_indice[..., None], batch_dims=3)
    #     # mask = tf.one_hot(gather_indice, VIRTUAL_DEPTH, on_value=True, off_value=False, dtype=tf.bool)
    #     # mask = tf.reduce_any(mask, axis=-2)
    #     # images_all = tf.where(mask[..., None], images_all, default_value)
    #     # images_nn = tf.pad(images_all, [[0, 0], [half_kernel[0], half_kernel[0]], [half_kernel[1], half_kernel[1]], [half_kernel[2], half_kernel[2]], [0, 0]],
    #     #                                                 mode="CONSTANT", constant_values=default_value)

    #     cost_grad_perturbation = tf.random.uniform([BATCH_SIZE, IMAGE_HEIGHT//test_strides[0], (IMAGE_WIDTH + test_strides[1] - 1)//test_strides[1], (IMAGE_DEPTH + test_strides[2] - 1)//test_strides[2], OUT_CHANNELS], dtype=images_all.dtype)
    #     # cost_grad_perturbation = tf.ones([BATCH_SIZE, IMAGE_HEIGHT//test_strides[0], (IMAGE_WIDTH + test_strides[1] - 1)//test_strides[1], (IMAGE_DEPTH + test_strides[2] - 1)//test_strides[2], OUT_CHANNELS], dtype=images_all.dtype)
    #     @tf.function
    #     def test_check(*args):
    #         cost = sparse_conv3d_fast(*args, base_plane, dilations=DILATIONS_SIZE, dynamic_default=True, strides=test_strides)
    #         return tf.reduce_sum(cost*cost_grad_perturbation)
    #     with self.cached_session():
    #         # res = sparse_conv2d(images, filters, base_plane, default_value, offsets)
    #         theoretical, numerical = gradient_checker_v2.compute_gradient(test_check, [images, filters, default_value])
    #         # err = gradient_checker_v2.max_error(theoretical, numerical)
    #     # print(images_all, filters)
    #     self.assertAllClose(theoretical[0], numerical[0])
    #     self.assertAllClose(theoretical[1], numerical[1])
    #     self.assertAllClose(theoretical[2], numerical[2])
    #     # self.assertAllClose(theoretical[3], numerical[3])

    # @parameterized.parameters(
    #   (1, 5, 1, 1, 1, 1, (1, 1, 1), (1, 1, 1)),
    #   (2, 64, 80, 16, 32, 32, (3, 3, 3), (1, 1, 1)),
    #   (2, 64, 80, 16, 20, 32, (3, 3, 3), (1, 1, 1)),
    # )
    # def testTransposeForward(self, BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, DILATIONS_SIZE):
    #     depth_factor = 2
    #     out_depth = IMAGE_DEPTH*depth_factor
    #     # tf.random.set_seed(np.random.randint(0, tf.int64.max))
    #     test_strides = (2, 2, 2)

    #     in_shape = np.array((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
    #     out_shape = (in_shape + np.array(test_strides) - 1)//np.array(test_strides)

    #     full_in_shape = np.array((IMAGE_HEIGHT, IMAGE_WIDTH, out_depth))
    #     full_out_shape = (full_in_shape + np.array(test_strides) - 1)//np.array(test_strides)

    #     images_all = tf.random.uniform([BATCH_SIZE, *full_out_shape, OUT_CHANNELS], dtype=tf.float32)
    #     filters = tf.random.uniform([*KERNEL_SIZE, IN_CHANNELS, OUT_CHANNELS], dtype=images_all.dtype)
    #     base_plane = tf.random.uniform([BATCH_SIZE, in_shape[0], in_shape[1], 1], minval=0, maxval=(full_in_shape[2] - in_shape[2] + 1), dtype=tf.int32)
    #     default_value = tf.zeros([], dtype=images_all.dtype)

    #     strided_base_plane = base_plane[:, 0::test_strides[0], 0::test_strides[1], :]
    #     strided_base_plane = (strided_base_plane + test_strides[2] - 1)//test_strides[2]
    #     gather_indice = strided_base_plane + np.arange(0, (IMAGE_DEPTH + test_strides[2] - 1)//test_strides[2], dtype=np.int32)[None, None, None, :]
    #     images = tf.gather_nd(images_all, gather_indice[..., None], batch_dims=3)
    #     mask = tf.one_hot(gather_indice, full_out_shape[2], on_value=True, off_value=False, dtype=tf.bool)
    #     mask = tf.reduce_any(mask, axis=-2)
    #     images_all = tf.where(mask[..., None], images_all, default_value)

    #     start = time.time()
    #     res = sparse_conv3d_transpose_fast(images, filters, default_value, base_plane, in_shape.astype(np.int32), dilations=DILATIONS_SIZE, strides=test_strides)
    #     my_time = time.time() - start


    #     partial = (full_out_shape - 1) * test_strides + 1 - full_in_shape
    #     pad_left = np.multiply(np.array(KERNEL_SIZE)//2,np.array(DILATIONS_SIZE))
    #     pad_right = np.maximum((np.array(KERNEL_SIZE) - 1)*np.array(DILATIONS_SIZE) - pad_left + partial, np.array([0, 0, 0]))
    #     pad_size = np.stack([pad_left, pad_right], axis=-1)
    #     # pad_size = np.concatenate([np.zeros((1, 2)), pad_size, np.zeros((1, 2))], axis=0)
    #     # images_nn = tf.pad(images_all, pad_size, mode="CONSTANT", constant_values=default_value)
    #     images_nn = images_all
    #     # print(pad_left, pad_right, pad_size)
    #     start = time.time()
    #     res_nn = tf.nn.conv3d_transpose(images_nn, filters, np.concatenate([[BATCH_SIZE,], full_in_shape + np.sum(pad_size, axis=1), [IN_CHANNELS,]], axis=0), strides=(1, *test_strides, 1), padding='VALID', dilations=(1, *DILATIONS_SIZE, 1))
    #     nn_time = time.time() - start

    #     gather_indice = base_plane + np.arange(0, IMAGE_DEPTH, dtype=np.int32)[None, None, None, :]
    #     # print(tf.shape(res_nn), tf.shape(gather_indice))
    #     res_nn = res_nn[:, pad_size[0, 0]:pad_size[0, 0] + full_in_shape[0], pad_size[1, 0]:pad_size[1, 0] + full_in_shape[1], pad_size[2, 0]:pad_size[2, 0] + full_in_shape[2], :]
    #     res_nn = tf.gather_nd(res_nn, gather_indice[..., None], batch_dims=3)

    #     # # print(tf.shape(res), tf.shape(res_nn), tf.shape(gather_indice))
    #     # # print(res, res_nn)
    #     # # print(images_all, base_plane, filters)
    #     # # print("my ", my_time/1000, " nn ", nn_time/1000)
    #     # # test_out = tf.reduce_sum(images[:, None, :KERNEL_SIZE[0], :KERNEL_SIZE[1], :KERNEL_SIZE[2], :]*filters[None, ...], axis=(2, 3, 4, 5))
    #     # # print(test_out, base_plane[:, :KERNEL_SIZE[0], :KERNEL_SIZE[1]])
    #     # # print(res[:, half_kernel[0], half_kernel[1],  half_kernel[2], :])
    #     # # print(res_nn[:, half_kernel[0], half_kernel[1], half_kernel[2], :])
    #     # # print(res_nn, res, images_all, (base_plane[:, ::2, ::2, :] + 1)//2 )
    #     # print(images, filters)
    #     # print(res, res_nn)
    #     self.assertShapeEqual(res.numpy(), res_nn)
    #     self.assertAllClose(res, res_nn, rtol=1e-5)
      
    @parameterized.parameters(
      (1, 1, 1, 1, 1, 1, (3, 3, 3), (1, 1, 1)),
      (1, 8, 13, 6, 3, 4, (3, 3, 3), (1, 1, 1)),
      (2, 4, 6, 5, 2, 1, (5, 3, 3), (2, 2, 2)),
      (3, 4, 6, 4, 2, 3, (2, 2, 1), (2, 2, 2)),
    )
    def testTransposeGradientFloat64(self, BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, DILATIONS_SIZE):
        depth_factor = 2
        out_depth = IMAGE_DEPTH*depth_factor
        # tf.random.set_seed(np.random.randint(0, tf.int64.max))
        test_strides = (2, 2, 2)

        in_shape = np.array((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
        out_shape = (in_shape + np.array(test_strides) - 1)//np.array(test_strides)

        full_in_shape = np.array((IMAGE_HEIGHT, IMAGE_WIDTH, out_depth))
        full_out_shape = (full_in_shape + np.array(test_strides) - 1)//np.array(test_strides)

        images_all = tf.random.uniform([BATCH_SIZE, *full_out_shape, OUT_CHANNELS], dtype=tf.float64)
        filters = tf.random.uniform([*KERNEL_SIZE, IN_CHANNELS, OUT_CHANNELS], dtype=images_all.dtype)
        base_plane = tf.random.uniform([BATCH_SIZE, in_shape[0], in_shape[1], 1], minval=0, maxval=(full_in_shape[2] - in_shape[2] + 1), dtype=tf.int32)
        default_value = tf.zeros([], dtype=images_all.dtype)

        strided_base_plane = base_plane[:, 0::test_strides[0], 0::test_strides[1], :]
        strided_base_plane = (strided_base_plane + test_strides[2] - 1)//test_strides[2]
        gather_indice = strided_base_plane + np.arange(0, (IMAGE_DEPTH + test_strides[2] - 1)//test_strides[2], dtype=np.int32)[None, None, None, :]
        images = tf.gather_nd(images_all, gather_indice[..., None], batch_dims=3)
        mask = tf.one_hot(gather_indice, full_out_shape[2], on_value=True, off_value=False, dtype=tf.bool)
        mask = tf.reduce_any(mask, axis=-2)
        images_all = tf.where(mask[..., None], images_all, default_value)

        cost_grad_perturbation = tf.random.uniform([BATCH_SIZE, *in_shape, IN_CHANNELS], dtype=images_all.dtype)
        # cost_grad_perturbation = tf.ones([BATCH_SIZE, IMAGE_HEIGHT//test_strides[0], (IMAGE_WIDTH + test_strides[1] - 1)//test_strides[1], (IMAGE_DEPTH + test_strides[2] - 1)//test_strides[2], OUT_CHANNELS], dtype=images_all.dtype)
        @tf.function
        def test_check(*args):
            cost = sparse_conv3d_transpose_fast(*args, base_plane, in_shape.astype(np.int32), dilations=DILATIONS_SIZE, dynamic_default=True, strides=test_strides)
            return tf.reduce_sum(cost*cost_grad_perturbation)
        with self.cached_session():
            # res = sparse_conv2d(images, filters, base_plane, default_value, offsets)
            theoretical, numerical = gradient_checker_v2.compute_gradient(test_check, [images, filters, default_value])
            # err = gradient_checker_v2.max_error(theoretical, numerical)
        # print(images_all, filters)
        self.assertAllClose(theoretical[0], numerical[0])
        self.assertAllClose(theoretical[1], numerical[1])
        # self.assertAllClose(theoretical[2], numerical[2])
        # self.assertAllClose(theoretical[3], numerical[3])

if __name__ == "__main__":
  test.main()