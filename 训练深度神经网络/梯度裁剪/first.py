from tensorflow import keras
import sklearn

# 在反向传播期间裁剪梯度 是它们永远不会超过某个阈值 最常用于RNN 因为RNN难以批量归一化
# 对于其他网络BN就够了通常


# 在model.compile时进行 设置优化器的梯度阈值
optimizer = keras.optimizers.SGD(clipvalue=1.0)
# 该优化器将梯度向量每个分量都裁剪为-1.0 ~ 1.0 之间
# [0.9,100] => [0.9,1.0]


# 如果要确保不会修改梯度向量的方向 用clipnorm
optimizer = keras.optimizers.SGD(clipnorm=1.0)
# [0.9,100] => [0.00899964,0.9999595]
print(sklearn.__version__)
