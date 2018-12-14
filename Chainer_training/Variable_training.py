# coding: UTF-8

# ライブラリのインポート
import chainer
import numpy as np
from chainer import Variable

# numpyの配列を生成
input_array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
print(input_array)

# Variableオブジェクトを作成
x = Variable(input_array)
print(x.data)

# 計算
# yもVariableオブジェクト
y = x * 2 + 1
print(y.data)