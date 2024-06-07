import math

import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt


# a_kの定義
def a_k(k, TCT_lambda, TCT_mu):
    X = (TCT_mu / (TCT_lambda + TCT_mu)) * (TCT_lambda / (TCT_lambda + TCT_mu)) ** k
    return X


# 到着率
# TCT_lambda = 0.7336929
# サービス率
TCT_mu = 1 / 12.09
# 最大バッチサイズ
# b = 1000
# 平均サービス時間
ES = 12.09
# 平均サービス時間の２次モーメント
ESS = 2 / (TCT_mu * TCT_mu)
# 最大バッチサイズに対するTCT
TCT = []
# グラフのx軸
x = []
# 占有率0.5
x5 = []
y5 = []
# 占有率0.6
x6 = []
y6 = []
# 占有率0.7
x7 = []
y7 = []
# 占有率0.8
x8 = []
y8 = []
# 占有率0.9
x9 = []
y9 = []
# 占有率
a = 0.5


# 優先順位なし
for i in np.arange(1, 75, 1):
    if i>= 74:
        for j in np.arange(74, 74.25, 0.05):
            TCT_lambda = j
            # 最大バッチサイズ
            b = 900

            sum_f = 0
            EW = 0
            ET = 0

            for k in range(0, b):
                pk = a_k(k, TCT_lambda, TCT_mu)
                f = pk * TCT_mu * (TCT_lambda * ESS * (b - k) + ES * (b * (b - 1) - k * (k - 1)) + 2 * b * ES * (b - k))
                sum_f += f

            EW = (1 / (2 * TCT_lambda * (b - TCT_lambda * ES))) * (
                    TCT_lambda * TCT_lambda * ESS - b * (b - 1) - 2 * (b - TCT_lambda * ES) ** 2 + sum_f)

            # 平均滞在時間ET
            ET = EW + ES
            print(ET)
            x.append(TCT_lambda)
            TCT.append(ET)
    else:
        TCT_lambda = i
        # 最大バッチサイズ
        b = 900

        sum_f = 0
        EW = 0
        ET = 0

        for k in range(0, b):
            pk = a_k(k, TCT_lambda, TCT_mu)
            f = pk * TCT_mu * (TCT_lambda * ESS * (b - k) + ES * (b * (b - 1) - k * (k - 1)) + 2 * b * ES * (b - k))
            sum_f += f

        EW = (1 / (2 * TCT_lambda * (b - TCT_lambda * ES))) * (
                TCT_lambda * TCT_lambda * ESS - b * (b - 1) - 2 * (b - TCT_lambda * ES) ** 2 + sum_f)

        # 平均滞在時間ET
        ET = EW + ES
        print(ET)
        x.append(TCT_lambda)
        TCT.append(ET)



#優先順位あり
for i in np.arange(1, 149, 1):
    if i >= 147:
        for j in np.arange(147, 148.6, 0.2):
            TCT_lambda = j * a
            # 最大バッチサイズ
            b = 900

            sum_f = 0
            EW = 0
            ET = 0

            for k in range(0, b):
                pk = a_k(k, TCT_lambda, TCT_mu)
                f = pk * TCT_mu * (TCT_lambda * ESS * (b - k) + ES * (b * (b - 1) - k * (k - 1)) + 2 * b * ES * (b - k))
                sum_f += f

            EW = (1 / (2 * TCT_lambda * (b - TCT_lambda * ES))) * (
                        TCT_lambda * TCT_lambda * ESS - b * (b - 1) - 2 * (b - TCT_lambda * ES) ** 2 + sum_f)

            # 平均滞在時間ET
            ET = EW + ES
            print(ET)

            x5.append(j)
            y5.append(ET)
    else:
        TCT_lambda = i * a
        # 最大バッチサイズ
        b = 900

        sum_f = 0
        EW = 0
        ET = 0

        for k in range(0, b):
            pk = a_k(k, TCT_lambda, TCT_mu)
            f = pk * TCT_mu * (TCT_lambda * ESS * (b - k) + ES * (b * (b - 1) - k * (k - 1)) + 2 * b * ES * (b - k))
            sum_f += f

        EW = (1 / (2 * TCT_lambda * (b - TCT_lambda * ES))) * (
                TCT_lambda * TCT_lambda * ESS - b * (b - 1) - 2 * (b - TCT_lambda * ES) ** 2 + sum_f)

        # 平均滞在時間ET
        ET = EW + ES
        print(ET)

        x5.append(i)
        y5.append(ET)


a = 0.6
#優先順位あり
for i in np.arange(1, 124, 1):
    if i >= 123:
        for j in np.arange(123, 123.69, 0.05):
            TCT_lambda = j * a
            # 最大バッチサイズ
            b = 900

            sum_f = 0
            EW = 0
            ET = 0

            for k in range(0, b):
                pk = a_k(k, TCT_lambda, TCT_mu)
                f = pk * TCT_mu * (TCT_lambda * ESS * (b - k) + ES * (b * (b - 1) - k * (k - 1)) + 2 * b * ES * (b - k))
                sum_f += f

            EW = (1 / (2 * TCT_lambda * (b - TCT_lambda * ES))) * (
                        TCT_lambda * TCT_lambda * ESS - b * (b - 1) - 2 * (b - TCT_lambda * ES) ** 2 + sum_f)

            # 平均滞在時間ET
            ET = EW + ES
            print(ET)

            x6.append(j)
            y6.append(ET)
    else:
        TCT_lambda = i * a
        # 最大バッチサイズ
        b = 900

        sum_f = 0
        EW = 0
        ET = 0

        for k in range(0, b):
            pk = a_k(k, TCT_lambda, TCT_mu)
            f = pk * TCT_mu * (TCT_lambda * ESS * (b - k) + ES * (b * (b - 1) - k * (k - 1)) + 2 * b * ES * (b - k))
            sum_f += f

        EW = (1 / (2 * TCT_lambda * (b - TCT_lambda * ES))) * (
                TCT_lambda * TCT_lambda * ESS - b * (b - 1) - 2 * (b - TCT_lambda * ES) ** 2 + sum_f)

        # 平均滞在時間ET
        ET = EW + ES
        print(ET)

        x6.append(i)
        y6.append(ET)

a = 0.7
#優先順位あり
for i in np.arange(1, 106, 1):
    if i >= 105:
        for j in np.arange(105, 106.1, 0.1):
            TCT_lambda = j * a
            # 最大バッチサイズ
            b = 900

            sum_f = 0
            EW = 0
            ET = 0

            for k in range(0, b):
                pk = a_k(k, TCT_lambda, TCT_mu)
                f = pk * TCT_mu * (TCT_lambda * ESS * (b - k) + ES * (b * (b - 1) - k * (k - 1)) + 2 * b * ES * (b - k))
                sum_f += f

            EW = (1 / (2 * TCT_lambda * (b - TCT_lambda * ES))) * (
                        TCT_lambda * TCT_lambda * ESS - b * (b - 1) - 2 * (b - TCT_lambda * ES) ** 2 + sum_f)

            # 平均滞在時間ET
            ET = EW + ES
            print(ET)

            x7.append(j)
            y7.append(ET)
    else:
        TCT_lambda = i * a
        # 最大バッチサイズ
        b = 900

        sum_f = 0
        EW = 0
        ET = 0

        for k in range(0, b):
            pk = a_k(k, TCT_lambda, TCT_mu)
            f = pk * TCT_mu * (TCT_lambda * ESS * (b - k) + ES * (b * (b - 1) - k * (k - 1)) + 2 * b * ES * (b - k))
            sum_f += f

        EW = (1 / (2 * TCT_lambda * (b - TCT_lambda * ES))) * (
                TCT_lambda * TCT_lambda * ESS - b * (b - 1) - 2 * (b - TCT_lambda * ES) ** 2 + sum_f)

        # 平均滞在時間ET
        ET = EW + ES
        print(ET)

        x7.append(i)
        y7.append(ET)

a = 0.8
#優先順位あり
for i in np.arange(1, 93, 1):
    if i >= 92:
        for j in np.arange(92, 92.8, 0.05):
            TCT_lambda = j * a
            # 最大バッチサイズ
            b = 900

            sum_f = 0
            EW = 0
            ET = 0

            for k in range(0, b):
                pk = a_k(k, TCT_lambda, TCT_mu)
                f = pk * TCT_mu * (TCT_lambda * ESS * (b - k) + ES * (b * (b - 1) - k * (k - 1)) + 2 * b * ES * (b - k))
                sum_f += f

            EW = (1 / (2 * TCT_lambda * (b - TCT_lambda * ES))) * (
                        TCT_lambda * TCT_lambda * ESS - b * (b - 1) - 2 * (b - TCT_lambda * ES) ** 2 + sum_f)

            # 平均滞在時間ET
            ET = EW + ES
            print(ET)

            x8.append(j)
            y8.append(ET)
    else:
        TCT_lambda = i * a
        # 最大バッチサイズ
        b = 900

        sum_f = 0
        EW = 0
        ET = 0

        for k in range(0, b):
            pk = a_k(k, TCT_lambda, TCT_mu)
            f = pk * TCT_mu * (TCT_lambda * ESS * (b - k) + ES * (b * (b - 1) - k * (k - 1)) + 2 * b * ES * (b - k))
            sum_f += f

        EW = (1 / (2 * TCT_lambda * (b - TCT_lambda * ES))) * (
                TCT_lambda * TCT_lambda * ESS - b * (b - 1) - 2 * (b - TCT_lambda * ES) ** 2 + sum_f)

        # 平均滞在時間ET
        ET = EW + ES
        print(ET)

        x8.append(i)
        y8.append(ET)

a = 0.9
#優先順位あり
for i in np.arange(1, 83, 1):
    if i >= 82:
        for j in np.arange(82, 82.5, 0.05):
            TCT_lambda = j * a
            # 最大バッチサイズ
            b = 900

            sum_f = 0
            EW = 0
            ET = 0

            for k in range(0, b):
                pk = a_k(k, TCT_lambda, TCT_mu)
                f = pk * TCT_mu * (TCT_lambda * ESS * (b - k) + ES * (b * (b - 1) - k * (k - 1)) + 2 * b * ES * (b - k))
                sum_f += f

            EW = (1 / (2 * TCT_lambda * (b - TCT_lambda * ES))) * (
                        TCT_lambda * TCT_lambda * ESS - b * (b - 1) - 2 * (b - TCT_lambda * ES) ** 2 + sum_f)

            # 平均滞在時間ET
            ET = EW + ES
            print(ET)

            x9.append(j)
            y9.append(ET)
    else:
        TCT_lambda = i * a
        # 最大バッチサイズ
        b = 900

        sum_f = 0
        EW = 0
        ET = 0

        for k in range(0, b):
            pk = a_k(k, TCT_lambda, TCT_mu)
            f = pk * TCT_mu * (TCT_lambda * ESS * (b - k) + ES * (b * (b - 1) - k * (k - 1)) + 2 * b * ES * (b - k))
            sum_f += f

        EW = (1 / (2 * TCT_lambda * (b - TCT_lambda * ES))) * (
                TCT_lambda * TCT_lambda * ESS - b * (b - 1) - 2 * (b - TCT_lambda * ES) ** 2 + sum_f)

        # 平均滞在時間ET
        ET = EW + ES
        print(ET)

        x9.append(i)
        y9.append(ET)

# plt.scatter(x, TCT, s = 5,label='No priority')
# plt.scatter(x9, y9, s = 5,label='priority α=0.9')
# plt.scatter(x8, y8, s = 5,label='priority α=0.8')
# plt.scatter(x7, y7, s = 5,label='priority α=0.7')
# plt.scatter(x6, y6, s = 5,label='priority α=0.6')
# plt.scatter(x5, y5, s = 5,label='priority α=0.5')
plt.plot(x, TCT, label = 'No Priority')
plt.plot(x9, y9, label = 'Priority $α = 0.9$')
plt.plot(x8, y8, label = 'Priority $α = 0.8$')
plt.plot(x7, y7, label = 'Priority $α = 0.7$')
plt.plot(x6, y6, label = 'Priority $α = 0.6$')
plt.plot(x5, y5, label = 'Priority $α = 0.5$')

plt.xlabel("Arrival rate of all transactions $\lambda$")
plt.ylabel("E[$T_{\mathrm{H}}$] [s]")
plt.legend()
plt.yscale("log")
plt.ylim(0, 5000)
plt.show()
