from tensorflow.python.platform import test
from custom_helper_op import sparse_conv2d
import numpy as np

class SparseConv2DTest(test.TestCase):
    @parameterized.parameters(
      (1, 10, 20, 30, 20),
      (1, 10, 20, 30, 20),
    )
    def testForward(self, BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IN_CHANNELS, OUT_CHANNELS):
        images = np.random.random([BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IN_CHANNELS])
        sparse_conv2d()
        self.assertAllClose(np_relu6, tf_relu6)
        self.assertShapeEqual(np_relu6, tf_relu6)

    def testGradientFloat64(self):
        with self.cached_session():
            x = np.asarray(
                [[-0.9, -0.7, -0.5, -0.3, -0.1], [6.1, 6.3, 6.5, 6.7, 6.9]],
                dtype=np.float64,
                order="F")
            err = gradient_checker_v2.max_error(
                *gradient_checker_v2.compute_gradient(nn_ops.relu6, [x]))
        self.assertLess(err, 1e-10)

if __name__ == "__main__":
  test.main()