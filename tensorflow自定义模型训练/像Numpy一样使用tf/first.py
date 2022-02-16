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
print(tf.strings.length(p, unit="UTF8_CHAR"))
# tf.Tensor([4 6 5 2], shape=(4,), dtype=int32)
r = tf.strings.unicode_decode(p, "UTF8")  # 将字符转为UTF-8编码
print("r>>", r)


# Ragged tensors 不规则张量
print(r[1])
# tf.Tensor([ 67 111 102 102 101 101], shape=(6,), dtype=int32)
print(r[1:3])
# <tf.RaggedTensor [[67, 111, 102, 102, 101, 101], [99, 97, 102, 102, 232]]>
r2 = tf.ragged.constant([[65, 66], [], [67]])
print(tf.concat([r, r2], axis=0))
# <tf.RaggedTensor [[67, 97, 102, 233], [67, 111, 102, 102, 101, 101], [99, 97, 102, 102, 232], [21654, 21857], [65, 66], [], [67]]>
r3 = tf.ragged.constant([[68, 69, 70], [71], [], [72, 73]])
print(tf.concat([r, r3], axis=1))  # 将r3的每一行最佳到r响应的行
# <tf.RaggedTensor [[67, 97, 102, 233, 68, 69, 70], [67, 111, 102, 102, 101, 101, 71], [99, 97, 102, 102, 232], [21654, 21857, 72, 73]]>
print("r3UTF-8", tf.strings.unicode_encode(r3, "UTF-8"))
#tf.Tensor([b'DEF' b'G' b'' b'HI'], shape=(4,), dtype=string)
print(r.to_tensor())
''' 当有缺省项时 默认填充0处理
[[   67    97   102   233     0     0]
 [   67   111   102   102   101   101]
 [   99    97   102   102   232     0]
 [21654 21857     0     0     0     0]], shape=(4, 6), dtype=int32)
'''


# Sparse tensors 稀疏张量
s = tf.SparseTensor(indices=[[0, 1], [1, 0], [2, 3]],
                    values=[1., 2., 3.],
                    dense_shape=[3, 4])
print(s)
'''
SparseTensor(indices=tf.Tensor(
[[0 1]
 [1 0]
 [2 3]], shape=(3, 2), dtype=int64), values=tf.Tensor([1. 2. 3.], shape=(3,), dtype=float32), dense_shape=tf.Tensor([3 4], shape=(2,), dtype=int64))
'''


print("dense>>", tf.sparse.to_dense(s))  # 稀疏矩阵转为稠密矩阵
'''
tf.Tensor(
[[0. 1. 0. 0.]
 [2. 0. 0. 0.]
 [0. 0. 0. 3.]], shape=(3, 4), dtype=float32)
'''
# 数乘 values 每个元素与2.0相乘
print(s * 2.0)
'''
SparseTensor(indices=tf.Tensor(
[[0 1]
 [1 0]
 [2 3]], shape=(3, 2), dtype=int64), values=tf.Tensor([2. 4. 6.], shape=(3,), dtype=float32), dense_shape=tf.Tensor([3 4], shape=(2,), dtype=int64))
'''


s4 = tf.constant([[10., 20.], [30., 40.], [50., 60.], [70., 80.]])
print("sparse_dense_matmul", tf.sparse.sparse_dense_matmul(s, s4))  # 稀疏与稠密矩阵相乘
'''
tf.Tensor(
[[ 30.  40.]
 [ 20.  40.]
 [210. 240.]], shape=(3, 2), dtype=float32)
'''
# 稀疏矩阵排列 行优先排列
print("reorder", tf.sparse.reorder(tf.SparseTensor(indices=[[0, 1], [2, 0], [1, 3]],
                                                   values=[1., 2., 3.],
                                                   dense_shape=[3, 4])))
'''
SparseTensor(indices=tf.Tensor(
[[0 1]
 [1 3]
 [2 0]], shape=(3, 2), dtype=int64), values=tf.Tensor([1. 3. 2.], shape=(3,), dtype=float32), dense_shape=tf.Tensor([3 4], shape=(2,), dtype=int64))
'''


# Sets 集合
set1 = tf.constant([[2, 3, 5, 7], [7, 9, 0, 0]])
set2 = tf.constant([[4, 5, 6], [9, 10, 0]])
# 并集
print(tf.sparse.to_dense(tf.sets.union(set1, set2)))
# 差集
print(tf.sparse.to_dense(tf.sets.difference(set1, set2)))
# 交集
print(tf.sparse.to_dense(tf.sets.intersection(set1, set2)))
'''
tf.Tensor(
[[ 2  3  4  5  6  7]
 [ 0  7  9 10  0  0]], shape=(2, 6), dtype=int32) 第二行是 0 7 9 10 后两位空
tf.Tensor(
[[2 3 7]
 [7 0 0]], shape=(2, 3), dtype=int32)
tf.Tensor(
[[5 0]
 [0 9]], shape=(2, 2), dtype=int32)
'''


# TF Variables 使用tf.constant定义常量 使用tf.Variables定义变量
# 其提供var.assign() 可以进行重新赋值操作
v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
v.assign(2 * v)
# 可以使用索引对特定元素进行赋值
v[0, 1].assign(42)
v[:, 2].assign([0., 1.])
print("v", v)
'''
<tf.Variable 'Variable:0' shape=(2, 3) dtype=float32, numpy=
array([[ 2., 42.,  0.],
       [ 8., 10.,  1.]], dtype=float32)>
'''
try:
    v[1] = [7., 8., 9.]
except TypeError as ex:
    print("TypeError", ex)
    print("不支持直接使用=进行赋值，理论上仍是常量，但提供assign方法")
# 根据下标进行元素替换 var.scatter_nd_update
v.scatter_nd_update(indices=[[0, 0], [1, 2]],
                    updates=[100., 200.])
print("scatter_nd_update", v)
'''
<tf.Variable 'Variable:0' shape=(2, 3) dtype=float32, numpy=
array([[100.,  42.,   0.],
       [  8.,  10., 200.]], dtype=float32)>
'''


# 使用索引切片对元素进行替换
sparse_delta = tf.IndexedSlices(values=[[1., 2., 3.], [4., 5., 6.]],
                                indices=[1, 0])
print("tf.IndexedSlices", sparse_delta)
# tf.IndexedSlices IndexedSlices(indices=[1, 0], values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
v.scatter_update(sparse_delta)
print("scatter_update", v)
'''
scatter_update <tf.Variable 'Variable:0' shape=(2, 3) dtype=float32, numpy=
array([[4., 5., 6.],
       [1., 2., 3.]], dtype=float32)>
'''


# Tensor Arrays 张量数组
array = tf.TensorArray(dtype=tf.float32, size=3)
# 下标写入
array = array.write(0, tf.constant([1., 2.]))
array = array.write(1, tf.constant([3., 10.]))
array = array.write(2, tf.constant([5., 7.]))
# 下标索引
print(array.read(1))
'''
tf.Tensor([ 3. 10.], shape=(2,), dtype=float32)
'''
# array.stack()
print(array.stack())
'''
tf.Tensor(
[[1. 2.]
 [0. 0.]
 [5. 7.]], shape=(3, 2), dtype=float32)
'''
mean, variance = tf.nn.moments(array.stack(), axes=0)
print("均值", mean)
print("方差", variance)
'''
均值 tf.Tensor([2. 3.], shape=(2,), dtype=float32)
方差 tf.Tensor([4.6666665 8.666667 ], shape=(2,), dtype=float32)
'''
