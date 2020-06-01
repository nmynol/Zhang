# 是否归一化
import numpy as np
from scipy import optimize as opt


def normalizing_input_data(coor_data):
    x_avg = np.mean(coor_data[:, 0])
    y_avg = np.mean(coor_data[:, 1])
    sx = np.sqrt(2) / np.std(coor_data[:, 0])   # 加不加根号2结果一样
    sy = np.sqrt(2) / np.std(coor_data[:, 1])

    norm_matrix = np.matrix([[sx, 0, -sx * x_avg],
                             [0, sy, -sy * y_avg],
                             [0, 0, 1]])
    return norm_matrix


def value(H, pic_coor, real_coor):
    Y = np.array([])
    for i in range(len(real_coor)):
        single_real_coor = np.array([real_coor[i, 0], real_coor[i, 1], 1])
        U = np.dot(H.reshape(3, 3), single_real_coor)
        U /= U[-1]
        Y = np.append(Y, U[:2])

    Y_NEW = (pic_coor.reshape(-1) - Y)

    return Y_NEW    # (108, )


def jacobian(H, pic_coor, real_coor):
    J = []
    for i in range(len(real_coor)):
        sx = H[0] * real_coor[i][0] + H[1] * real_coor[i][1] + H[2]
        sy = H[3] * real_coor[i][0] + H[4] * real_coor[i][1] + H[5]
        w = H[6] * real_coor[i][0] + H[7] * real_coor[i][1] + H[8]
        w2 = w * w

        J.append(np.array([real_coor[i][0] / w, real_coor[i][1] / w, 1 / w,
                           0, 0, 0,
                           -sx * real_coor[i][0] / w2, -sx * real_coor[i][1] / w2, -sx / w2]))

        J.append(np.array([0, 0, 0,
                           real_coor[i][0] / w, real_coor[i][1] / w, 1 / w,
                           -sy * real_coor[i][0] / w2, -sy * real_coor[i][1] / w2, -sy / w2]))

    return np.array(J)


def get_homography(pic, real):
    sum_H = []

    for i in range(len(pic)):
        M = []
        # # 构造标准化矩阵
        # pic_norm_mat = normalizing_input_data(pic[i])
        # real_norm_mat = normalizing_input_data(real[i])
        for j in range(len(pic[i])):
            # 转齐次坐标
            hom_pic = np.array([pic[i][j][0], pic[i][j][1], 1])
            hom_real = np.array([real[i][j][0], real[i][j][1], 1])
            # # 标准化
            # hom_pic = np.dot(pic_norm_mat, hom_pic)
            # hom_real = np.dot(real_norm_mat, hom_real)
            # hom_pic = np.array(hom_pic).squeeze()
            # hom_real = np.array(hom_real).squeeze()
            # 构造M矩阵
            M.append(np.array([
                hom_real[0], hom_real[1], 1, 0, 0, 0, -hom_pic[0] * hom_real[0], -hom_pic[0] * hom_real[1], -hom_pic[0]
            ]))
            M.append(np.array([
                0, 0, 0, hom_real[0], hom_real[1], 1, -hom_pic[1] * hom_real[0], -hom_pic[1] * hom_real[1], -hom_pic[1]
            ]))
        # 最小二乘
        U, S, V = np.linalg.svd((np.array(M, dtype='float')).reshape((-1, 9)))
        H = V[-1].reshape((3, 3))

        # # 去归一化
        # H = np.dot(np.dot(np.linalg.inv(pic_norm_mat), H), real_norm_mat)

        H /= H[-1, -1]
        # LM微调
        H = np.array(H)
        '''
        从这点开始有迷之差别，不知道为啥
        '''
        final_H = opt.leastsq(value, H, Dfun=jacobian, args=(pic[i], real[i]))[0]
        final_H /= np.array(final_H[-1])

        sum_H.append(final_H)

    return np.array(sum_H)
