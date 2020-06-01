import numpy as np


def get_extrinsics_param(H, in_param):
    sum_ex = []

    inv_in_param = np.linalg.inv(in_param)
    for i in range(len(H)):
        h0 = (H[i].reshape(3, 3))[:, 0]
        h1 = (H[i].reshape(3, 3))[:, 1]
        h2 = (H[i].reshape(3, 3))[:, 2]

        lam = 1 / np.linalg.norm(np.dot(inv_in_param, h0))

        r0 = lam * np.dot(inv_in_param, h0)
        r1 = lam * np.dot(inv_in_param, h1)
        t = lam * np.dot(inv_in_param, h2)
        r2 = np.cross(r0, r1)

        ex_param = np.array([r0, r1, r2, t]).transpose()
        sum_ex.append(ex_param)

    return sum_ex
