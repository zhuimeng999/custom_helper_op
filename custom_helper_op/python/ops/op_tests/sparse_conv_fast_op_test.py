import tensorflow as tf
from tensorflow.python.platform import test
from absl.testing import parameterized
from custom_helper_op import sparse_conv3d_fast, sparse_conv3d_transpose_fast
import numpy as np
from tensorflow.python.ops import gradient_checker_v2
import time


class SparseConv3DFastTest(test.TestCase, parameterized.TestCase):
    # @parameterized.parameters(
    #   (1, 34, 68, 8, 20, 32, 4, (1, 1, 1), (1, 1, 1)),
    #   (2, 64, 80, 16, 20, 32, 32, (3, 3, 3), (1, 1, 1)),
    #   (2, 64, 80, 16, 20, 32, 32, (3, 3, 3), (1, 1, 1)),
    # )
    # def testForward(self, BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, VIRTUAL_DEPTH, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, DILATIONS_SIZE):
    #     tf.random.set_seed(np.random.randint(0, tf.int64.max))
    #     images_all = tf.random.uniform([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, VIRTUAL_DEPTH, IN_CHANNELS], dtype=tf.float32)
    #     filters = tf.random.uniform([KERNEL_SIZE[0], KERNEL_SIZE[1], KERNEL_SIZE[2], IN_CHANNELS, OUT_CHANNELS], dtype=images_all.dtype)
    #     # filters = tf.ones([KERNEL_SIZE[0], KERNEL_SIZE[1], KERNEL_SIZE[2], IN_CHANNELS, OUT_CHANNELS], dtype=tf.float64)
    #     base_plane = tf.random.uniform([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 1], minval=0, maxval=(VIRTUAL_DEPTH - IMAGE_DEPTH + 1), dtype=tf.int32)
    #     # base_plane = (base_plane//2)*2
    #     # test_start_d = 5
    #     # base_plane = tf.ones([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 1], dtype=tf.int32) * test_start_d
    #     default_value = tf.random.uniform([], dtype=images_all.dtype)
    #     # default_value = tf.constant(0, dtype=images_all.dtype)

    #     half_kernel = np.array(KERNEL_SIZE)//2
    #     gather_indice = base_plane + np.arange(0, IMAGE_DEPTH, dtype=np.int32)[None, None, None, :]
    #     images = tf.gather_nd(images_all, gather_indice[..., None], batch_dims=3)
    #     mask = tf.one_hot(gather_indice, VIRTUAL_DEPTH, on_value=True, off_value=False, dtype=tf.bool)
    #     mask = tf.reduce_any(mask, axis=-2)
    #     images_all = tf.where(mask[..., None], images_all, default_value)
    #     pad_size = np.multiply(half_kernel,np.array(DILATIONS_SIZE))
    #     assert(len(pad_size) == 3)
    #     images_nn = tf.pad(images_all, [[0, 0], [pad_size[0], pad_size[0]], [pad_size[1], pad_size[1]], [pad_size[2], pad_size[2]], [0, 0]],
    #                                                     mode="CONSTANT", constant_values=default_value)
    #     start = time.time()
    #     res = sparse_conv3d_fast(images, filters, default_value, base_plane, dilations=DILATIONS_SIZE, strides=(1, 1, 1))

    #     my_time = time.time() - start

    #     # filters_nn = tf.transpose(filters, [1, 2, 3, 4, 0])
    #     start = time.time()
    #     # print("base_plane ", base_plane)
    #     # print(images_nn)
    #     res_nn = tf.nn.conv3d(images_nn, filters, strides=(1, 1, 1, 1, 1), padding="VALID", dilations=(1, *DILATIONS_SIZE, 1))
    #     # print(res_nn)
    #     nn_time = time.time() - start
    #     gather_indice = (base_plane[:, ::1, ::1, :] + 0)//1 + np.arange(0, IMAGE_DEPTH//1, dtype=np.int32)[None, None, None, :]
    #     # print(tf.shape(res_nn), tf.shape(res))
    #     # print('indice ', gather_indice)
    #     res_nn = tf.gather_nd(res_nn, gather_indice[..., None], batch_dims=3)
    #     # print("my ", my_time/1000, " nn ", nn_time/1000)
    #     # test_out = tf.reduce_sum(images[:, None, :KERNEL_SIZE[0], :KERNEL_SIZE[1], :KERNEL_SIZE[2], :]*filters[None, ...], axis=(2, 3, 4, 5))
    #     # print(test_out, base_plane[:, :KERNEL_SIZE[0], :KERNEL_SIZE[1]])
    #     # print(res[:, half_kernel[0], half_kernel[1],  half_kernel[2], :])
    #     # print(res_nn[:, half_kernel[0], half_kernel[1], half_kernel[2], :])
    #     # print(res_nn, res, images_all, (base_plane[:, ::2, ::2, :] + 1)//2 )
    #     self.assertShapeEqual(res.numpy(), res_nn)
    #     self.assertAllClose(res, res_nn)
      
    # @parameterized.parameters(
    #   (2, 6, 6, 8, 16, 3, 5, (3, 3, 3), (1, 1, 1)),
    #   (1, 8, 13, 6, 15, 3, 4, (3, 3, 3), (1, 1, 1)),
    #   (2, 4, 6, 5, 8, 2, 1, (5, 3, 3), (2, 2, 2)),
    #   (3, 4, 6, 4, 8, 2, 3, (2, 2, 1), (2, 2, 2)),
    # )
    # def testGradientFloat64(self, BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, VIRTUAL_DEPTH, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, DILATIONS_SIZE):
    #     test_strides = (2, 2, 2)

    #     tf.random.set_seed(np.random.randint(0, tf.int64.max))
    #     images_all = tf.random.uniform([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, VIRTUAL_DEPTH, IN_CHANNELS], dtype=tf.float64)
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

    @parameterized.parameters(
      (1, 5, 1, 1, 1, 1, (1, 1, 1), (1, 1, 1)),
      # (2, 64, 80, 16, 20, 32, 32, (3, 3, 3), (1, 1, 1)),
      # (2, 64, 80, 16, 20, 32, 32, (3, 3, 3), (1, 1, 1)),
    )
    def testTransposeForward(self, BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, DILATIONS_SIZE):
        depth_factor = 2
        tf.random.set_seed(np.random.randint(0, tf.int64.max))
        test_strides = np.array((1, 1, 1))
        in_shape = np.array((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH))
        out_shape = in_shape*test_strides
        images_all = tf.random.uniform([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH*depth_factor, IN_CHANNELS], dtype=tf.float32)
        filters = tf.random.uniform([*KERNEL_SIZE, IN_CHANNELS, OUT_CHANNELS], dtype=images_all.dtype)
        base_plane = tf.random.uniform([BATCH_SIZE, out_shape[0], out_shape[1], 1], minval=0, maxval=(IMAGE_DEPTH*depth_factor - IMAGE_DEPTH + 1), dtype=tf.int32)
        default_value = tf.random.uniform([], dtype=images_all.dtype)

        gather_indice = base_plane[:, ::test_strides[0], ::test_strides[1], :] + np.arange(0, IMAGE_DEPTH, dtype=np.int32)[None, None, None, :]
        images = tf.gather_nd(images_all, gather_indice[..., None], batch_dims=3)
        mask = tf.one_hot(gather_indice, IMAGE_DEPTH*depth_factor, on_value=True, off_value=False, dtype=tf.bool)
        mask = tf.reduce_any(mask, axis=-2)
        images_all = tf.where(mask[..., None], images_all, default_value)

        pad_left = np.multiply(np.array(KERNEL_SIZE)//2,np.array(DILATIONS_SIZE))
        pad_right = np.array(KERNEL_SIZE)*np.array(DILATIONS_SIZE) - pad_left - 1
        pad_size = np.stack([pad_left, pad_right], axis=-1)
        pad_size = np.concatenate([np.zeros((1, 2)), pad_size, np.zeros((1, 2))], axis=0)
        images_nn = tf.pad(images_all, pad_size, mode="CONSTANT", constant_values=default_value)
        # start = time.time()
        # res = sparse_conv3d_transpose_fast(images, filters, default_value, base_plane, dilations=DILATIONS_SIZE, strides=(2, 2, 2))

        # my_time = time.time() - start

        # # filters_nn = tf.transpose(filters, [1, 2, 3, 4, 0])
        # start = time.time()
        print(pad_left, pad_right, pad_size)
        full_size_shape = tf.shape(images_nn)[1:4] + (tf.shape(images_nn)[1:4] - 1)*(test_strides - 1)
        print(full_size_shape)
        res_nn = tf.nn.conv3d_transpose(images_nn, filters, np.concatenate([[BATCH_SIZE,], full_size_shape, [OUT_CHANNELS,]], axis=0), strides=(1, *test_strides, 1), padding="VALID", dilations=(1, *DILATIONS_SIZE, 1))
        print(tf.shape(res_nn))
        # nn_time = time.time() - start
        # gather_indice = (base_plane[:, ::1, ::1, :] + 0)//1 + np.arange(0, IMAGE_DEPTH//1, dtype=np.int32)[None, None, None, :]
        # # print(tf.shape(res_nn), tf.shape(res))
        # # print('indice ', gather_indice)
        # res_nn = tf.gather_nd(res_nn, gather_indice[..., None], batch_dims=3)
        # # print("my ", my_time/1000, " nn ", nn_time/1000)
        # # test_out = tf.reduce_sum(images[:, None, :KERNEL_SIZE[0], :KERNEL_SIZE[1], :KERNEL_SIZE[2], :]*filters[None, ...], axis=(2, 3, 4, 5))
        # # print(test_out, base_plane[:, :KERNEL_SIZE[0], :KERNEL_SIZE[1]])
        # # print(res[:, half_kernel[0], half_kernel[1],  half_kernel[2], :])
        # # print(res_nn[:, half_kernel[0], half_kernel[1], half_kernel[2], :])
        # # print(res_nn, res, images_all, (base_plane[:, ::2, ::2, :] + 1)//2 )
        # self.assertShapeEqual(res.numpy(), res_nn)
        # self.assertAllClose(res, res_nn)
      
    # @parameterized.parameters(
    #   (2, 6, 6, 8, 16, 3, 5, (3, 3, 3), (1, 1, 1)),
    #   (1, 8, 13, 6, 15, 3, 4, (3, 3, 3), (1, 1, 1)),
    #   (2, 4, 6, 5, 8, 2, 1, (5, 3, 3), (2, 2, 2)),
    #   (3, 4, 6, 4, 8, 2, 3, (2, 2, 1), (2, 2, 2)),
    # )
    # def testTransposeGradientFloat64(self, BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, VIRTUAL_DEPTH, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, DILATIONS_SIZE):
    #     test_strides = (2, 2, 2)

    #     tf.random.set_seed(np.random.randint(0, tf.int64.max))
    #     images_all = tf.random.uniform([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, VIRTUAL_DEPTH, IN_CHANNELS], dtype=tf.float64)
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

    #     cost_grad_perturbation = tf.random.uniform([BATCH_SIZE, IMAGE_HEIGHT*test_strides[0], IMAGE_WIDTH*test_strides[1], IMAGE_DEPTH*test_strides[2], OUT_CHANNELS], dtype=images_all.dtype)
    #     # cost_grad_perturbation = tf.ones([BATCH_SIZE, IMAGE_HEIGHT//test_strides[0], (IMAGE_WIDTH + test_strides[1] - 1)//test_strides[1], (IMAGE_DEPTH + test_strides[2] - 1)//test_strides[2], OUT_CHANNELS], dtype=images_all.dtype)
    #     @tf.function
    #     def test_check(*args):
    #         cost = sparse_conv3d_transpose_fast(*args, base_plane, dilations=DILATIONS_SIZE, dynamic_default=True, strides=test_strides)
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

if __name__ == "__main__":
  test.main()