# coding: UTF-8

# ライブラリのインポート
import chainer
import numpy as np
from chainer import Variable, Chain, optimizers
import chainer.links as L
import chainer.functions as F

# モデルをクラスで記述
class MyChain(Chain):
    def __init__(self):
        super().__init__(
            l1 = L.Linear(1, 2),
            l2 = L.Linear(2, 1)
        )

    def __call__(self, x):
        h = F.sigmoid(self.l1(x))
        return self.l2(h)

# Optimizerの記述
model = MyChain()
optimizer = optimizers.SGD()    # 最適化アルゴリズム: 確率的降下法
optimizer = optimizers.Adam()    # 最適化アルゴリズム: 確率的降下法
optimizer.setup(model)

# Optimizerの実行
input_array = np.array([[1]], dtype=np.float32)
answer_array = np.array([[1]], dtype=np.float32)
x = Variable(input_array)
t = Variable(answer_array)

model.cleargrads()
y = model(x)
loss = F.mean_squared_error(y, t)
loss.backward()

print(model.l1.W.data)
optimizer.update()
print(model.l1.W.data)