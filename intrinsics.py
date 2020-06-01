# 两种计算方法
import numpy as np


def create_v(p, q, H):
    H = H.reshape(3, 3)
    return np.array([
        H[0, p] * H[0, q], H[0, p] * H[1, q] + H[1, p] * H[0, q], H[1, p] * H[1, q],
        H[2, p] * H[0, q] + H[0, p] * H[2, q], H[2, p] * H[1, q] + H[1, p] * H[2, q], H[2, p] * H[2, q]
    ])


def get_intrinsics_param(H):
    V = np.array([])
    # 构造V矩阵
    for i in range(len(H)):
        V = np.append(V, np.array([create_v(0, 1, H[i])]))
        V = np.append(V, np.array([create_v(0, 0, H[i]) - create_v(1, 1, H[i])]))

    U, S, V = np.linalg.svd((np.array(V, dtype='float')).reshape((-1, 6)))
    b = V[-1]

    # w = b[0] * b[2] * b[5] - b[1] * b[1] * b[5] - b[0] * b[4] * b[4] + 2 * b[1] * b[3] * b[4] - b[2] * b[3] * b[3]
    # d = b[0] * b[2] - b[1] * b[1]
    # alpha = np.sqrt(w / (d * b[0]))
    # beta = np.sqrt(w / d ** 2 * b[0])
    # gamma = np.sqrt(w / (d ** 2 * b[0])) * b[1]
    # u0 = (b[1] * b[4] - b[2] * b[3]) / d
    # v0 = (b[1] * b[3] - b[0] * b[4]) / d

    v0 = (b[1] * b[3] - b[0] * b[4]) / (b[0] * b[2] - b[1] * b[1])
    lam = b[5] - (b[3] * b[3] + v0 * (b[1] * b[3] - b[0] * b[4])) / (b[0])
    alpha = np.sqrt(lam / b[0])
    beta = np.sqrt((lam * b[0]) / (b[0] * b[2] - b[1] * b[1]))
    gamma = -b[1] * alpha * alpha * beta / lam
    u0 = gamma * v0 / beta - b[3] * alpha * alpha / lam

    return np.array([
        [alpha, gamma, u0],
        [0, beta, v0],
        [0, 0, 1]
    ])
