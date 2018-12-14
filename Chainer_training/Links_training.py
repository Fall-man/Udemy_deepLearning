# coding: UTF-8

# ライブラリのインポート
import chainer
import numpy as np
from chainer import Variable
import chainer.links as L

# Linear Link
l = L.Linear(3, 4)
print(l.W.data)
print(l.b.data)

# オブジェクトlによりyを計算
# input_array = np.array([[1, 2, 3]], dtype=np.float32)
input_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
x = Variable(input_array)
y = l(x)
print(y.data)

# lの勾配を0に初期化
l.cleargrads()

# y->lに遡って微分の計算
y.grad = np.ones((2, 4), dtype=np.float32)
y.backward()
print(l.W.grad)
print(l.b.grad)