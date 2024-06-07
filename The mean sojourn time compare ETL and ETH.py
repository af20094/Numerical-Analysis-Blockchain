import numpy as np
import numpy.linalg
import matplotlib.pyplot as plt


# A_l行列（b*b)の定義
def A_l(b, a_k, a_kk):
    X = np.eye(b, None, 1)
    X = X * a_k
    X[b - 1, 0] = a_kk
    return X


# a_kの定義
def a_k(k, TCT_lambda, TCT_mu):
    X = (TCT_mu / (TCT_lambda + TCT_mu)) * (TCT_lambda / (TCT_lambda + TCT_mu)) ** k
    return X


# a_kkの定義
def a_kk(k, TCT_lambda, TCT_mu):
    X = (TCT_lambda / (TCT_lambda + TCT_mu)) * (TCT_mu / (TCT_lambda + TCT_mu)) ** (k + 1)
    return X


# bar_a_kの定義
def bar_a_k(k, TCT_lambda, TCT_mu):
    X = 0
    for i in range(0, k + 1):
        X += a_k(i, TCT_lambda, TCT_mu)

    result = 1 - X
    return result


# bar_a_kkの定義
def bar_a_kk(k, TCT_lambda, TCT_mu):
    X = 0
    for i in range(0, k + 2):
        X += a_k(i, TCT_lambda, TCT_mu)

    result = 1 - X
    return result


# B_k(b*1)行列の定義
def B_k(b, bar_a_k, bar_a_kk):
    X = np.zeros((b, 1))
    for i in range(0, b):
        X[i, 0] = bar_a_k

    X[b - 1, 0] = bar_a_kk

    return X


# C_0(1*b)行列の定義
def C_0(b, a_0):
    X = np.zeros((1, b))
    X[0, 0] = a_0

    return X


# 1の行列（b*1）
def one(b):
    X = np.zeros((1, b))
    for i in range(0, b):
        X[0, i] = 1

    X = np.transpose(X)
    return X


# 再帰関数R_nの定義
def calculate_R(n, b, A, R_1):
    if n == 0:
        return np.eye(b)

    I = np.eye(b, None)  # 単位行列
    R_n_minus_1 = R_1
    sum_term = np.zeros((b, b))

    for k in range(2, 100):
        sum_term += np.dot(np.linalg.matrix_power(R_n_minus_1, k), A[k])

    sum_term += A[0]
    result = np.dot(sum_term, np.linalg.inv(I - A[1]))
    return result


# R^(l - 1) * A_l の総和
def sigma_R_A(R, A, b):
    result = np.zeros((b, b))
    for l in range(1, b):
        result += np.dot(np.linalg.matrix_power(R, l - 1), A[l])

    return result


# R^(l - 1) * B_l の総和
def sigma_R_B(R, B, b):
    result = np.zeros((b, b))
    for l in range(1, 100):
        result += np.dot(np.linalg.matrix_power(R, l - 1), B[l])

    return result


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
# グラフのx軸2
x2 = []
# グラフのy軸
y = []
# グラフのy軸2
y2 = []
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
a = 0.9

# EWL　占有率0.5
L5 = []
X5 = []
# EWL　占有率0.6
L6 = []
X6 = []
# EWL　占有率0.7
L7 = []
X7 = []
# EWL　占有率0.8
L8 = []
X8 = []
# EWL　占有率0.9
L9 = []
X9 = []

# for文抜け
finish = 0

# その他の情報の平均滞在時間の閾値betaを満たす被災情報の平均滞在時間に対する最適な占有率alpha*

# その他の情報の平均滞在時間閾値 [s]
#beta = 600

# 最適な占有率alpha*の導出
for i in np.arange(1, 74, 1):
    if i >= 74:
        for j in np.arange(74, 75, 0.4):
            TCT_lambda = j * a
            # 最大バッチサイズ
            b = 900
            sum_f = 0
            EW = 0
            ET = 0
            EWL = 0
            EWall = 0
            ETL = 0

            for k in range(0, b):
                pk = a_k(k, TCT_lambda, TCT_mu)
                f = pk * TCT_mu * (TCT_lambda * ESS * (b - k) + ES * (b * (b - 1) - k * (k - 1)) + 2 * b * ES * (b - k))
                sum_f += f

            EW = (1 / (2 * TCT_lambda * (b - TCT_lambda * ES))) * (
                        TCT_lambda * TCT_lambda * ESS - b * (b - 1) - 2 * (b - TCT_lambda * ES) ** 2 + sum_f)

            # 平均滞在時間ET
            ET = EW + ES


            sum_f = 0
            for k in range(0, b):
                pk = a_k(k, j, TCT_mu)
                f = pk * TCT_mu * (j * ESS * (b - k) + ES * (b * (b - 1) - k * (k - 1)) + 2 * b * ES * (b - k))
                sum_f += f
            EWall = (1 / (2 * j * (b - j * ES))) * (
                    j * j * ESS - b * (b - 1) - 2 * (b - j * ES) ** 2 + sum_f)

            EWL = j / (j - TCT_lambda) * EWall - TCT_lambda / (j - TCT_lambda) * EW
            ETL = EWL + ES

            if ETL > beta:
                # lambda_H*を決定
                for gamma in np.arange(0, 1, 0.05):
                    lambda_Hstar = TCT_lambda * (1 - gamma)
                    # 最大バッチサイズ
                    b = 900
                    sum_f = 0
                    EW = 0
                    ET = 0
                    EWL = 0
                    EWall = 0
                    ETL = 0

                    for k in range(0, b):
                        pk = a_k(k, lambda_Hstar, TCT_mu)
                        f = pk * TCT_mu * (
                                    lambda_Hstar * ESS * (b - k) + ES * (b * (b - 1) - k * (k - 1)) + 2 * b * ES * (
                                        b - k))
                        sum_f += f

                    EW = (1 / (2 * lambda_Hstar * (b - lambda_Hstar * ES))) * (
                            lambda_Hstar * lambda_Hstar * ESS - b * (b - 1) - 2 * (b - lambda_Hstar * ES) ** 2 + sum_f)

                    # 平均滞在時間ET
                    ET = EW + ES

                    sum_f = 0
                    for k in range(0, b):
                        pk = a_k(k, j, TCT_mu)
                        f = pk * TCT_mu * (j * ESS * (b - k) + ES * (b * (b - 1) - k * (k - 1)) + 2 * b * ES * (b - k))
                        sum_f += f
                    EWall = (1 / (2 * j * (b - j * ES))) * (
                            j * j * ESS - b * (b - 1) - 2 * (b - j * ES) ** 2 + sum_f)

                    EWL = j / (j - lambda_Hstar) * EWall - lambda_Hstar / (j - lambda_Hstar) * EW
                    ETL = EWL + ES

                    x.append(j)
                    y.append(ETL)

    else:
            TCT_lambda = i * a
            # 最大バッチサイズ
            b = 900
            sum_f = 0
            EW = 0
            ET = 0
            EWL = 0
            EWall = 0
            ETL =0


            for k in range(0, b):
                pk = a_k(k, TCT_lambda, TCT_mu)
                f = pk * TCT_mu * (TCT_lambda * ESS * (b - k) + ES * (b * (b - 1) - k * (k - 1)) + 2 * b * ES * (b - k))
                sum_f += f

            EW = (1 / (2 * TCT_lambda * (b - TCT_lambda * ES))) * (
                    TCT_lambda * TCT_lambda * ESS - b * (b - 1) - 2 * (b - TCT_lambda * ES) ** 2 + sum_f)

            # 平均滞在時間ET
            ET = EW + ES


            sum_f = 0
            for k in range(0, b):
                pk = a_k(k, i, TCT_mu)
                f = pk * TCT_mu * (i * ESS * (b - k) + ES * (b * (b - 1) - k * (k - 1)) + 2 * b * ES * (b - k))
                sum_f += f
            EWall = (1 / (2 * i * (b - i * ES))) * (
                    i * i * ESS - b * (b - 1) - 2 * (b - i * ES) ** 2 + sum_f)

            EWL = (i / (i - TCT_lambda)) * EWall - (TCT_lambda / (i - TCT_lambda)) * EW
            ETL = EWL + ES

            x.append(i)
            y.append(ETL)



# 制御前の被災情報の平均滞在時間
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


# all
for i in np.arange(1, 75, 1):
    if i >= 74:
        for j in np.arange(74, 74.3, 0.2):
            TCT_lambda = j * a
            # 最大バッチサイズ
            b = 900
            sum_f = 0
            EW = 0
            ET = 0
            EWL = 0
            EWall = 0
            ETL = 0

            for k in range(0, b):
                pk = a_k(k, TCT_lambda, TCT_mu)
                f = pk * TCT_mu * (TCT_lambda * ESS * (b - k) + ES * (b * (b - 1) - k * (k - 1)) + 2 * b * ES * (b - k))
                sum_f += f

            EW = (1 / (2 * TCT_lambda * (b - TCT_lambda * ES))) * (
                        TCT_lambda * TCT_lambda * ESS - b * (b - 1) - 2 * (b - TCT_lambda * ES) ** 2 + sum_f)

            # 平均滞在時間ET
            ET = EW + ES


            sum_f = 0
            for k in range(0, b):
                pk = a_k(k, j, TCT_mu)
                f = pk * TCT_mu * (j * ESS * (b - k) + ES * (b * (b - 1) - k * (k - 1)) + 2 * b * ES * (b - k))
                sum_f += f
            EWall = (1 / (2 * j * (b - j * ES))) * (
                    j * j * ESS - b * (b - 1) - 2 * (b - j * ES) ** 2 + sum_f)

            x2.append(j)
            y2.append(EWall + ES)

    else:
            TCT_lambda = i * a
            # 最大バッチサイズ
            b = 900
            sum_f = 0
            EW = 0
            ET = 0
            EWL = 0
            EWall = 0
            ETL =0


            for k in range(0, b):
                pk = a_k(k, TCT_lambda, TCT_mu)
                f = pk * TCT_mu * (TCT_lambda * ESS * (b - k) + ES * (b * (b - 1) - k * (k - 1)) + 2 * b * ES * (b - k))
                sum_f += f

            EW = (1 / (2 * TCT_lambda * (b - TCT_lambda * ES))) * (
                    TCT_lambda * TCT_lambda * ESS - b * (b - 1) - 2 * (b - TCT_lambda * ES) ** 2 + sum_f)

            # 平均滞在時間ET
            ET = EW + ES


            sum_f = 0
            for k in range(0, b):
                pk = a_k(k, i, TCT_mu)
                f = pk * TCT_mu * (i * ESS * (b - k) + ES * (b * (b - 1) - k * (k - 1)) + 2 * b * ES * (b - k))
                sum_f += f
            EWall = (1 / (2 * i * (b - i * ES))) * (
                    i * i * ESS - b * (b - 1) - 2 * (b - i * ES) ** 2 + sum_f)

            x2.append(i)
            y2.append(EWall + ES)







#plt.scatter(x, y, s = 5,label='$\mathrm{E}[T_L]$')
#plt.scatter(x2, y2, s = 5, label="$\mathrm{E}[T]$")
#plt.scatter(x2, y2, s = 5, label ="The mean sojourn time [s]")
#plt.scatter(x9, y9, s = 5,label='$\mathrm{E}[T_H]$')
#plt.scatter(X9, L9, s = 5,label='priority L class α=0.9')
plt.plot(x, y, label = '$\mathrm{E}[T_L]$')
#plt.plot(x2, y2, label = '$\mathrm{E}[T]$')
plt.plot(x9, y9, label = '$\mathrm{E}[T_H]$')

plt.xlabel("Arrival rate of all transactions $\lambda$")
#plt.ylabel("Optimal Value gamma ")
plt.ylabel("The mean sojourn time $[s]$")
plt.legend()
plt.yscale("log")
plt.ylim(0, 5000)
plt.show()
