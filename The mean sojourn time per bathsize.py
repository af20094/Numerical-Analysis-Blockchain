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

# 遷移確率（X_n+1=j,X_n=i)
# 0要素をすべて0にするコード　np.zeros(b,b)
# result =  a * np.eye(n)  aは定数、nは行列の長さ
# a_k=(TCT_lambda/(TCT_lambda+TCT_mu)) * ((TCT_mu/(TCT_lambda+TCT_mu))^k
# bar_a_k=1-sum(a_i,i,1,k)
# if isinstance((j-i-1)/b, int)==true
# p_ij=a_(j-i-1)/b
# else if j==0
# p_ij=-a_int(i/b)+1
# else
# p_ij=0

for i in np.arange(303, 310, 1):
    TCT_lambda = 50 * a
    # 最大バッチサイズ
    b = i
    # 空のリスト（A_l)
    A = []
    # 空のリスト（R）
    R = []
    # 空のリスト（B_k)
    B = []
    # 空のリスト（pi_n)ベクトル
    pi = []
    # 単位行列b*b
    I = np.eye(b)
    # 1の行列b*1
    O = one(b)
    # B_0
    B_0 = bar_a_k(1, TCT_lambda, TCT_mu)
    # a_0
    a_0 = a_k(0, TCT_lambda, TCT_mu)

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
#優先順位あり
for i in np.arange(310, 901, 10):
    TCT_lambda = 50 * a
    # 最大バッチサイズ
    b = i
    # 空のリスト（A_l)
    A = []
    # 空のリスト（R）
    R = []
    # 空のリスト（B_k)
    B = []
    # 空のリスト（pi_n)ベクトル
    pi = []
    # 単位行列b*b
    I = np.eye(b)
    # 1の行列b*1
    O = one(b)
    # B_0
    B_0 = bar_a_k(1, TCT_lambda, TCT_mu)
    # a_0
    a_0 = a_k(0, TCT_lambda, TCT_mu)

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
for i in np.arange(364, 370, 1):
    TCT_lambda = 50 * a
    # 最大バッチサイズ
    b = i
    # 空のリスト（A_l)
    A = []
    # 空のリスト（R）
    R = []
    # 空のリスト（B_k)
    B = []
    # 空のリスト（pi_n)ベクトル
    pi = []
    # 単位行列b*b
    I = np.eye(b)
    # 1の行列b*1
    O = one(b)
    # B_0
    B_0 = bar_a_k(1, TCT_lambda, TCT_mu)
    # a_0
    a_0 = a_k(0, TCT_lambda, TCT_mu)

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

#優先順位あり
for i in np.arange(370, 901, 10):
    TCT_lambda = 50 * a
    # 最大バッチサイズ
    b = i
    # 空のリスト（A_l)
    A = []
    # 空のリスト（R）
    R = []
    # 空のリスト（B_k)
    B = []
    # 空のリスト（pi_n)ベクトル
    pi = []
    # 単位行列b*b
    I = np.eye(b)
    # 1の行列b*1
    O = one(b)
    # B_0
    B_0 = bar_a_k(1, TCT_lambda, TCT_mu)
    # a_0
    a_0 = a_k(0, TCT_lambda, TCT_mu)

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
for i in np.arange(425, 430, 1):
    TCT_lambda = 50 * a
    # 最大バッチサイズ
    b = i
    # 空のリスト（A_l)
    A = []
    # 空のリスト（R）
    R = []
    # 空のリスト（B_k)
    B = []
    # 空のリスト（pi_n)ベクトル
    pi = []
    # 単位行列b*b
    I = np.eye(b)
    # 1の行列b*1
    O = one(b)
    # B_0
    B_0 = bar_a_k(1, TCT_lambda, TCT_mu)
    # a_0
    a_0 = a_k(0, TCT_lambda, TCT_mu)

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

#優先順位あり
for i in np.arange(430, 901, 10):
    TCT_lambda = 50 * a
    # 最大バッチサイズ
    b = i
    # 空のリスト（A_l)
    A = []
    # 空のリスト（R）
    R = []
    # 空のリスト（B_k)
    B = []
    # 空のリスト（pi_n)ベクトル
    pi = []
    # 単位行列b*b
    I = np.eye(b)
    # 1の行列b*1
    O = one(b)
    # B_0
    B_0 = bar_a_k(1, TCT_lambda, TCT_mu)
    # a_0
    a_0 = a_k(0, TCT_lambda, TCT_mu)

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
for i in np.arange(486, 490, 1):
    TCT_lambda = 50 * a
    # 最大バッチサイズ
    b = i
    # 空のリスト（A_l)
    A = []
    # 空のリスト（R）
    R = []
    # 空のリスト（B_k)
    B = []
    # 空のリスト（pi_n)ベクトル
    pi = []
    # 単位行列b*b
    I = np.eye(b)
    # 1の行列b*1
    O = one(b)
    # B_0
    B_0 = bar_a_k(1, TCT_lambda, TCT_mu)
    # a_0
    a_0 = a_k(0, TCT_lambda, TCT_mu)

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

#優先順位あり
for i in np.arange(490, 901, 10):
    TCT_lambda = 50 * a
    # 最大バッチサイズ
    b = i
    # 空のリスト（A_l)
    A = []
    # 空のリスト（R）
    R = []
    # 空のリスト（B_k)
    B = []
    # 空のリスト（pi_n)ベクトル
    pi = []
    # 単位行列b*b
    I = np.eye(b)
    # 1の行列b*1
    O = one(b)
    # B_0
    B_0 = bar_a_k(1, TCT_lambda, TCT_mu)
    # a_0
    a_0 = a_k(0, TCT_lambda, TCT_mu)

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
for i in np.arange(547, 550, 1):
    TCT_lambda = 50 * a
    # 最大バッチサイズ
    b = i
    # 空のリスト（A_l)
    A = []
    # 空のリスト（R）
    R = []
    # 空のリスト（B_k)
    B = []
    # 空のリスト（pi_n)ベクトル
    pi = []
    # 単位行列b*b
    I = np.eye(b)
    # 1の行列b*1
    O = one(b)
    # B_0
    B_0 = bar_a_k(1, TCT_lambda, TCT_mu)
    # a_0
    a_0 = a_k(0, TCT_lambda, TCT_mu)

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

#優先順位あり
for i in np.arange(550, 901, 10):
    TCT_lambda = 50 * a
    # 最大バッチサイズ
    b = i
    # 空のリスト（A_l)
    A = []
    # 空のリスト（R）
    R = []
    # 空のリスト（B_k)
    B = []
    # 空のリスト（pi_n)ベクトル
    pi = []
    # 単位行列b*b
    I = np.eye(b)
    # 1の行列b*1
    O = one(b)
    # B_0
    B_0 = bar_a_k(1, TCT_lambda, TCT_mu)
    # a_0
    a_0 = a_k(0, TCT_lambda, TCT_mu)

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


# plt.scatter(x5, y5, s = 5,label='priority α=0.5')
# plt.scatter(x6, y6, s = 5,label='priority α=0.6')
# plt.scatter(x7, y7, s = 5,label='priority α=0.7')
# plt.scatter(x8, y8, s = 5,label='priority α=0.8')
# plt.scatter(x9, y9, s = 5,label='priority α=0.9')

plt.plot(x5, y5, label='priority α=0.5')
plt.plot(x6, y6, label='priority α=0.6')
plt.plot(x7, y7, label='priority α=0.7')
plt.plot(x8, y8, label='priority α=0.8')
plt.plot(x9, y9, label='priority α=0.9')


plt.xlabel("Maximum batch size b")
plt.ylabel("The mean sojourn time [s]")
plt.yscale("log")
plt.ylim(0, 2500)
plt.legend()
plt.show()
