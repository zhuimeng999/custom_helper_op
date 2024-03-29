import tensorflow as tf
import pytest
import sys
from tensorflow_addons.layers.equi_conv import EquiConv
from tensorflow_addons.utils import test_utils


@test_utils.run_all_in_graph_and_eager_modes
class EquiConvTest(tf.test.TestCase):
    def testKerasNHWC(self):
        channel = 32
        input = tf.ones(shape=[1, 10, 10, channel])
        layer = EquiConv(
            channel, (3, 3), 1, 1, (1, 1), 1, False, "same", "channels_last"
        )
        res = layer(input)
        self.assertAllEqual(tf.shape(input), tf.shape(res))

    def testKerasNCHW(self):
        channel = 32
        input = tf.ones(shape=[1, channel, 10, 10])
        layer = EquiConv(
            channel, (3, 3), 1, 1, (1, 1), 1, False, "same", "channels_first"
        )
        res = layer(input)
        self.assertAllEqual(tf.shape(input), tf.shape(res))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
