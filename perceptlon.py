# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 09:44:32 2020

@author: IKM1YH

perceptlon
"""

import numpy as np
import matplotlib.pyplot as plt
import random

# データ点の個数
N = 100

# データ点のために乱数列を固定
np.random.seed(0)

# ランダムな N×2 行列を生成 = 2次元空間上のランダムな点 N 個
X = np.random.randn(N, 2)

def h(x, y):
  return 5 * x + 3 * y - 1  #  真の分離平面 5x + 3y = 1

T = np.array([ 1 if h(x, y) > 0 else -1 for x, y in X])

# 特徴関数
def phi(x, y):
  return np.array([x, y, 1])

w = np.zeros(3)  # パラメータを初期化(3次の 0 ベクトル)

np.random.seed() # 乱数を初期化
while True:
  list = range(N)
  random.shuffle(list)

  misses = 0 # 予測を外した回数
  for n in list:
    x_n, y_n = X[n, :]
    t_n = T[n]

    # 予測
    predict = np.sign((w * phi(x_n, y_n)).sum())

    # 予測が不正解なら，パラメータを更新する
    if predict != t_n:
      w += t_n * phi(x_n, y_n)
      misses += 1

  # 予測が外れる点が無くなったら学習終了(ループを抜ける)
  if misses == 0:
    break

# 図を描くための準備
seq = np.arange(-3, 3, 0.02)
xlist, ylist = np.meshgrid(seq, seq)
zlist = [np.sign((w * phi(x, y)).sum()) for x, y in zip(xlist, ylist)]

# 分離平面と散布図を描画
plt.pcolor(xlist, ylist, zlist, alpha=0.2, edgecolors='white')
plt.plot(X[T== 1,0], X[T== 1,1], 'o', color='red')
plt.plot(X[T==-1,0], X[T==-1,1], 'o', color='blue')
plt.show()