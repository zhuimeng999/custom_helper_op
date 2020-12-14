import tensorflow as tf
from tensorflow.python.platform import test
from absl.testing import parameterized
from custom_helper_op import sparse_conv3d_fast
import numpy as np
from tensorflow.python.ops import gradient_checker_v2
import time


class SparseConv3DFastTest(test.TestCase, parameterized.TestCase):
    @parameterized.parameters(
      (1, 34, 68, 28, 46, 32, 4, (1, 1, 1), (1, 1, 1)),
      (2, 128, 160, 16, 48, 5, 6, (5, 3, 3), (1, 1, 1)),
      (2, 128, 160, 16, 48, 5, 6, (3, 3, 3), (2, 2, 2)),
    )
    def testForward(self, BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, VIRTUAL_DEPTH, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, DILATIONS_SIZE):
        images_all = tf.random.uniform([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, VIRTUAL_DEPTH, IN_CHANNELS], dtype=tf.float64)
        # filters = tf.random.uniform([KERNEL_SIZE[0], KERNEL_SIZE[1], KERNEL_SIZE[2], IN_CHANNELS, OUT_CHANNELS], dtype=tf.float64)
        filters = tf.ones([KERNEL_SIZE[0], KERNEL_SIZE[1], KERNEL_SIZE[2], IN_CHANNELS, OUT_CHANNELS], dtype=tf.float64)
        base_plane = tf.random.uniform([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 1], minval=0, maxval=(VIRTUAL_DEPTH - IMAGE_DEPTH + 1), dtype=tf.int32)
        # base_plane = (base_plane//2)*2
        # test_start_d = 5
        # base_plane = tf.ones([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 1], dtype=tf.int32) * test_start_d
        default_value = tf.random.uniform([], dtype=images_all.dtype)
        # default_value = tf.constant(0, dtype=images_all.dtype)

        half_kernel = np.array(KERNEL_SIZE)//2
        gather_indice = base_plane + np.arange(0, IMAGE_DEPTH, dtype=np.int32)[None, None, None, :]
        images = tf.gather_nd(images_all, gather_indice[..., None], batch_dims=3)
        mask = tf.one_hot(gather_indice, VIRTUAL_DEPTH, on_value=True, off_value=False, dtype=tf.bool)
        mask = tf.reduce_any(mask, axis=-2)
        images_all = tf.where(mask[..., None], images_all, default_value)
        pad_size = np.multiply(half_kernel,np.array(DILATIONS_SIZE))
        assert(len(pad_size) == 3)
        images_nn = tf.pad(images_all, [[0, 0], [pad_size[0], pad_size[0]], [pad_size[1], pad_size[1]], [pad_size[2], pad_size[2]], [0, 0]],
                                                        mode="CONSTANT", constant_values=default_value)
        start = time.time()
        res = sparse_conv3d_fast(images, filters, default_value, base_plane, dilations=DILATIONS_SIZE, strides=(2, 2, 2))

        my_time = time.time() - start

        # filters_nn = tf.transpose(filters, [1, 2, 3, 4, 0])
        start = time.time()
        # print("base_plane ", base_plane)
        # print(images_nn)
        res_nn = tf.nn.conv3d(images_nn, filters, strides=(1, 2, 2, 2, 1), padding="VALID", dilations=(1, *DILATIONS_SIZE, 1))
        # print(res_nn)
        nn_time = time.time() - start
        gather_indice = (base_plane[:, ::2, ::2, :] + 1)//2 + np.arange(0, IMAGE_DEPTH//2, dtype=np.int32)[None, None, None, :]
        # print(tf.shape(res_nn), tf.shape(res))
        # print('indice ', gather_indice)
        res_nn = tf.gather_nd(res_nn, gather_indice[..., None], batch_dims=3)
        # print("my ", my_time/1000, " nn ", nn_time/1000)
        # test_out = tf.reduce_sum(images[:, None, :KERNEL_SIZE[0], :KERNEL_SIZE[1], :KERNEL_SIZE[2], :]*filters[None, ...], axis=(2, 3, 4, 5))
        # print(test_out, base_plane[:, :KERNEL_SIZE[0], :KERNEL_SIZE[1]])
        # print(res[:, half_kernel[0], half_kernel[1],  half_kernel[2], :])
        # print(res_nn[:, half_kernel[0], half_kernel[1], half_kernel[2], :])
        # print(res_nn, res, images_all, (base_plane[:, ::2, ::2, :] + 1)//2 )
        self.assertShapeEqual(res.numpy(), res_nn)
        self.assertAllClose(res, res_nn)
      

    # @parameterized.parameters(
    #   (1, 10, 20, 3, 5, 4, 3, (3, 3, 3)),
    #   # (2, 4, 6, 5, 8, 2, 1, (5, 3, 3)),
    #   # (1, 128, 160, 32, 64, 4, 3, (3, 3, 3)),
    #   # (2, 128, 160, 32, 96, 30, 20, (5, 3, 3)),
    # )
    # def testGradTime(self, BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, VIRTUAL_DEPTH, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE):
    #     images_all = tf.random.uniform([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, VIRTUAL_DEPTH, IN_CHANNELS], dtype=tf.float32)
    #     filters = tf.random.uniform([KERNEL_SIZE[0], KERNEL_SIZE[1], KERNEL_SIZE[2], IN_CHANNELS, OUT_CHANNELS], dtype=tf.float32)
    #     base_plane = tf.random.uniform([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 1], minval=0, maxval=(VIRTUAL_DEPTH - IMAGE_DEPTH), dtype=tf.int32)
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
    #     images_nn = tf.pad(images_all, [[0, 0], [half_kernel[0], half_kernel[0]], [half_kernel[1], half_kernel[1]], [half_kernel[2], half_kernel[2]], [0, 0]],
    #                                                     mode="CONSTANT", constant_values=default_value)

    #     start = time.time()
    #     with tf.GradientTape() as tape:
    #       tape.watch([images, filters, default_value])
    #       res = sparse_conv3d(images, filters, default_value, base_plane)
    #       res = tf.reduce_mean(res)
    #     my_forward_time = time.time() - start
    #     start = time.time()
    #     grad = tape.gradient(res, [images, filters, default_value])
    #     my_backforward_time = time.time() - start

    #     start = time.time()
    #     with tf.GradientTape() as tape:
    #       tape.watch([images, filters])
    #       res_nn = tf.nn.conv3d(images, filters, strides=(1, 1, 1, 1, 1), padding="SAME")
    #       res = tf.reduce_mean(res_nn)
    #     nn_forward_time = time.time() - start
    #     nn_grad = tape.gradient(res, [images, filters])
    #     nn_backforward_time = time.time() - start

    #     print("my forward ", my_forward_time/1000, " my backfoward ", my_backforward_time/1000, " nn forward ", nn_forward_time/1000, " nn backforward ", nn_backforward_time/1000)
    #     self.assertShapeEqual(grad[0].numpy(), nn_grad[0])
    #     self.assertAllClose(grad[0], nn_grad[0])


    @parameterized.parameters(
      (1, 4, 8, 6, 6, 1, 1, (3, 3, 1), (1, 1, 1)),
      (1, 8, 13, 6, 15, 3, 4, (3, 3, 3), (1, 1, 1)),
      (2, 4, 6, 5, 8, 2, 1, (5, 3, 3), (2, 2, 2)),
    )
    def testGradientFloat64(self, BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, VIRTUAL_DEPTH, IN_CHANNELS, OUT_CHANNELS, KERNEL_SIZE, DILATIONS_SIZE):
        test_strides = (2, 2, 2)

        tf.random.set_seed(np.random.randint(0, tf.int64.max))
        images_all = tf.random.uniform([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, VIRTUAL_DEPTH, IN_CHANNELS], dtype=tf.float64)
        filters = tf.random.uniform([KERNEL_SIZE[0], KERNEL_SIZE[1], KERNEL_SIZE[2], IN_CHANNELS, OUT_CHANNELS], dtype=tf.float64)
        base_plane = tf.random.uniform([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 1], minval=0, maxval=(VIRTUAL_DEPTH - IMAGE_DEPTH + 1), dtype=tf.int32)
        # test_start_d = 5
        # base_plane = tf.ones([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, 1], dtype=tf.int32) * test_start_d
        default_value = tf.random.uniform([], dtype=images_all.dtype)
        # default_value = tf.constant(0, dtype=images_all.dtype)

        half_kernel = np.array(KERNEL_SIZE)//2
        gather_indice = base_plane + np.arange(0, IMAGE_DEPTH, dtype=np.int32)[None, None, None, :]
        images = tf.gather_nd(images_all, gather_indice[..., None], batch_dims=3)
        # mask = tf.one_hot(gather_indice, VIRTUAL_DEPTH, on_value=True, off_value=False, dtype=tf.bool)
        # mask = tf.reduce_any(mask, axis=-2)
        # images_all = tf.where(mask[..., None], images_all, default_value)
        # images_nn = tf.pad(images_all, [[0, 0], [half_kernel[0], half_kernel[0]], [half_kernel[1], half_kernel[1]], [half_kernel[2], half_kernel[2]], [0, 0]],
        #                                                 mode="CONSTANT", constant_values=default_value)

        cost_grad_perturbation = tf.random.uniform([BATCH_SIZE, IMAGE_HEIGHT//test_strides[0], (IMAGE_WIDTH + test_strides[1] - 1)//test_strides[1], (IMAGE_DEPTH + test_strides[2] - 1)//test_strides[2], OUT_CHANNELS], dtype=images_all.dtype)
        @tf.function
        def test_check(*args):
            cost = sparse_conv3d_fast(*args, base_plane, dilations=DILATIONS_SIZE, dynamic_default=True, strides=test_strides)
            return tf.reduce_sum(cost*cost_grad_perturbation)
        with self.cached_session():
            # res = sparse_conv2d(images, filters, base_plane, default_value, offsets)
            theoretical, numerical = gradient_checker_v2.compute_gradient(test_check, [images, filters, default_value])
            # err = gradient_checker_v2.max_error(theoretical, numerical)
        # print(images_all, filters)
        self.assertAllClose(theoretical[0], numerical[0])
        # self.assertAllClose(theoretical[1], numerical[1])
        # self.assertAllClose(theoretical[2], numerical[2])
        # self.assertAllClose(theoretical[3], numerical[3])

if __name__ == "__main__":
  test.main()