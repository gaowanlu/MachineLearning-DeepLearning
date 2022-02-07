import numpy as np
import tensorflow.keras as keras
from numpy import square
import tensorflow as tf
# 张量定义
m1 = tf.constant([
    [1, 2, 3],
    [4, 5, 6]
])
print(m1)
'''
tf.Tensor(
[[1 2 3]
 [4 5 6]], shape=(2, 3), dtype=int32)
'''
m2 = tf.constant(42.)
print(m2)  # tf.Tensor(42.0, shape=(), dtype=float32)
# shape、dtpye属性
print(m1.shape)  # (2, 3)
print(m2.dtype)  # <dtype: 'float32'>


# 索引
print(m1[:, 1:])
''' 所有行 从第2列到左后一列
tf.Tensor(
[[2 3]
 [5 6]], shape=(2, 2), dtype=int32)
'''
print(m1[:, 1, tf.newaxis])
'''
tf.Tensor(
[[2]
 [5]], shape=(2, 1), dtype=int32)
'''

# 张量操作
# + tf.add(t,10)
print(m1+10)
'''
tf.Tensor(
[[11 12 13]
 [14 15 16]], shape=(2, 3), dtype=int32)
'''
# tf.add() tf.multiply() tf.square() tf.exp() tf.sqrt()
# tf.reshape() tf.squeeze() tf.tile()
# tf.reduce_mean() tf.reduce_sum() tf.reduce_max() tf.math.log()
# np.mean()        np.sum()        np.max()        np.log()
# square 元素自平方
print(tf.square(m1))
'''
tf.Tensor(
[[ 1  4  9]
 [16 25 36]], shape=(2, 3), dtype=int32)
'''

# tf.transpose()转置  @矩阵相乘 tf.matmul
print(m1 @ tf.transpose(m1))
'''
tf.Tensor(
[[14 32]
 [32 77]], shape=(2, 2), dtype=int32)
'''


# 使用 keras.backend
K = keras.backend
print(K.square(K.transpose(m1))+10)
'''
tf.Tensor(
[[11 26]
 [14 35]
 [19 46]], shape=(3, 2), dtype=int32)
'''

# 张量与numpy之间的转换
a = np.array([1., 2., 3.])
print(tf.constant(a))
# tf.Tensor([1. 2. 3.], shape=(3,), dtype=float64)
print(tf.square(a))
# tf.Tensor([1. 4. 9.], shape=(3,), dtype=float64)
print(m1.numpy())  # 张量转numpy数组


# 张量元素类型转换
m3 = tf.constant([1, 2, 3])
print(tf.cast(m3, tf.float32).dtype)
# <dtype: 'float32'>


# 字符串支持
print(tf.constant(b"HELLO WORLD"))
print(tf.constant("HELLO WORLD").numpy())  # b'HELLO WORLD'
u = tf.constant([ord(c) for c in "你好"])
print(u)  # tf.Tensor([20320 22909], shape=(2,), dtype=int32)
# unicode支持
b = tf.strings.unicode_encode(u, "UTF-8")
print(tf.strings.length(b, unit="UTF8_CHAR"))
# tf.Tensor(2, shape=(), dtype=int32)
print(tf.strings.unicode_decode(b, "UTF-8"))
# tf.Tensor([20320 22909], shape=(2,), dtype=int32)

# String arrays
p = tf.constant(["Café", "Coffee", "caffè", "咖啡"])
tf.strings.length(p, unit="UTF8_CHAR")
r = tf.strings.unicode_decode(p, "UTF8")
r
print(r)


# Ragged tensors
print(r[1])
print(r[1:3])
r2 = tf.ragged.constant([[65, 66], [], [67]])
print(tf.concat([r, r2], axis=0))
r3 = tf.ragged.constant([[68, 69, 70], [71], [], [72, 73]])
print(tf.concat([r, r3], axis=1))
tf.strings.unicode_encode(r3, "UTF-8")
r.to_tensor()

# Sparse tensors
s = tf.SparseTensor(indices=[[0, 1], [1, 0], [2, 3]],
                    values=[1., 2., 3.],
                    dense_shape=[3, 4])
print(s)
tf.sparse.to_dense(s)
s2 = s * 2.0
try:
    s3 = s + 1.
except TypeError as ex:
    print(ex)
s4 = tf.constant([[10., 20.], [30., 40.], [50., 60.], [70., 80.]])
tf.sparse.sparse_dense_matmul(s, s4)
s5 = tf.SparseTensor(indices=[[0, 2], [0, 1]],
                     values=[1., 2.],
                     dense_shape=[3, 4])
print(s5)
try:
    tf.sparse.to_dense(s5)
except tf.errors.InvalidArgumentError as ex:
    print(ex)
s6 = tf.sparse.reorder(s5)
tf.sparse.to_dense(s6)


# Sets
set1 = tf.constant([[2, 3, 5, 7], [7, 9, 0, 0]])
set2 = tf.constant([[4, 5, 6], [9, 10, 0]])
tf.sparse.to_dense(tf.sets.union(set1, set2))
tf.sparse.to_dense(tf.sets.difference(set1, set2))
tf.sparse.to_dense(tf.sets.intersection(set1, set2))


# Variables
v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
v.assign(2 * v)
v[0, 1].assign(42)
v[:, 2].assign([0., 1.])
try:
    v[1] = [7., 8., 9.]
except TypeError as ex:
    print(ex)
v.scatter_nd_update(indices=[[0, 0], [1, 2]],
                    updates=[100., 200.])
sparse_delta = tf.IndexedSlices(values=[[1., 2., 3.], [4., 5., 6.]],
                                indices=[1, 0])
v.scatter_update(sparse_delta)


# Tensor Arrays
array = tf.TensorArray(dtype=tf.float32, size=3)
array = array.write(0, tf.constant([1., 2.]))
array = array.write(1, tf.constant([3., 10.]))
array = array.write(2, tf.constant([5., 7.]))
array.read(1)
array.stack()
mean, variance = tf.nn.moments(array.stack(), axes=0)
mean
variance
