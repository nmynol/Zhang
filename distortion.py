import numpy as np


def get_distortion(intrinsic_param, extrinsic_param, pic_coor, real_coor):
    D = []
    d = []
    for i in range(len(pic_coor)):
        for j in range(len(pic_coor[i])):
            # 转换为齐次坐标
            single_coor = np.array([real_coor[i][j][0], real_coor[i][j][1], 0, 1])

            # 利用现有内参及外参求出估计图像坐标
            u = np.dot(np.dot(intrinsic_param, extrinsic_param[i]), single_coor)
            [u_estim, v_estim] = [u[0] / u[2], u[1] / u[2]]     # 除的这个u[2]其实就是缩放系数

            # 求r
            coor_norm = np.dot(extrinsic_param[i], single_coor)
            coor_norm /= coor_norm[-1]  # 除的这个coor_norm[-1]其实就是缩放系数
            r = np.linalg.norm(coor_norm)

            # (u - u0) * r^2, (u - u0) * r^4
            D.append(np.array([(u_estim - intrinsic_param[0][2]) * r ** 2, (u_estim - intrinsic_param[0][2]) * r ** 4]))
            # (v - v0) * r^2, (v - v0) * r^4
            D.append(np.array([(v_estim - intrinsic_param[1][2]) * r ** 2, (v_estim - intrinsic_param[1][2]) * r ** 4]))

            # u^ - u
            d.append(pic_coor[i][j][0] - u_estim)
            # v^ - v
            d.append(pic_coor[i][j][1] - v_estim)
            '''
            D.append(np.array([(pic_coor[i][j, 0] - intrinsic_param[0, 2]) * r ** 2, (pic_coor[i][j, 0] - intrinsic_param[0, 2]) * r ** 4]))
            D.append(np.array([(pic_coor[i][j, 1] - intrinsic_param[1, 2]) * r ** 2, (pic_coor[i][j, 1] - intrinsic_param[1, 2]) * r ** 4]))
            #求出估计坐标与真实坐标的残差
            d.append(u_estim - pic_coor[i][j, 0])
            d.append(v_estim - pic_coor[i][j, 1])
            '''

    # 最小二乘法求解D * k = d
    D = np.array(D)
    k = np.dot(np.dot(np.linalg.inv(np.dot(D.T, D)), D.T), d)  # 用最小二乘那个公式

    '''
    #也可利用SVD求解D * k = d中的k
    U, S, Vh=np.linalg.svd(D, full_matrices=False)
    temp_S = np.array([[S[0], 0],
                       [0, S[1]]])
    temp_res = np.dot(Vh.transpose(), np.linalg.inv(temp_S))
    temp_res_res = np.dot(temp_res, U.transpose())
    k = np.dot(temp_res_res, d)
    '''
    return k
