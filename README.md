# 三维重建 tensorflow OP

- cost volume OP
在三维重建算法中，大量的神经网络，如MVS Net系列，会通过深度投影，构建cost volume，直接使用tensorflow提供的函数，需要大量的小型离散操作，极大地降低运行速度，并需要占用大量显存

经测试，cost volume能够提供三倍以上的运行速度，显存占用下降到原来的一半

- sparse convolution OP
三维重建流形卷积

正常的三维卷积在一个立方体内进行，但是在目前的三维重建普遍采用，从粗到精的网络结构，在上层不需要在整个cost volume空间中进行，而只需要在粗粒度提取的表面周围进行

- pfm 文件格式读取

tensorflow的读取pfm文件的函数，方便直接读取pfm文件
