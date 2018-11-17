# coding: utf-8

"""
単純パーセプトロンの実装

"""

import numpy as np
import matplotlib.pyplot as plt


rng = np.random.RandomState(123)

# dimention
D = 2 
# Neuron
N = 10
mean = 5


# make data
x1 = rng.randn(N, D) + np.array([0, 0])
x2 = rng.randn(N, D) + np.array([mean, mean])

"""print (x1)
#列の取り出し
print (x[:,1])"""

plt.plot(x1[:,0], x1[:,1], "o")
plt.plot(x2[:,0], x2[:,1], "o")

#plt.show()


# marge data
x = np.concatenate((x1, x2), axis=0)


# build model
w = np.zeros(D)
b = 0

def y(x):
    return step(np.dot(w, x) + b)

def step(x):
    #return 1 * (x >0)
    if x > 0:
        return 1
    else:
        return 0

# パラメータ更新式
# label
def t(i):
    if i < N:
        return 0
    else:
        return 1

while True:
    classified = True
    for i in range(N * 2):
        delta_w = (t(i) - y(x[i])) * x[i] 
        delta_b = (t(i) - y(x[i]))

        w += delta_w
        b += delta_b

        classified *= all(delta_w==0) * (delta_b==0)

    if classified:
        break

print ("w:", w)
print ("b:" , b)

print(w[0])
X1 = np.arange(0,10)
Y = (-b -w[0]*X1) / w[1]

plt.plot(Y)
plt.show()
