import cv2 as cv
import numpy as np
import os
from scipy import optimize as opt
import math
from homography import get_homography
from intrinsics import get_intrinsics_param
from extrinsics import get_extrinsics_param
from distortion import get_distortion
from refine_all import refinall_all_param


def extract_point(file_dir, row, column, unit):
    pic_name = os.listdir(file_dir)

    real_coor = np.zeros((row * column, 3))
    real_coor[:, :2] = np.mgrid[0:row, 0:column].T.reshape(-1, 2)
    real_coor *= unit

    for pic in pic_name:
        path = file_dir + '/' + pic
        img = cv.imread(path)

        succ, pic_coor = cv.findChessboardCorners(img, (row, column), None)

        if succ:
            pic_coor = pic_coor.reshape(-1, 2)
            pic_points.append(pic_coor)

            real_points.append(real_coor)
            real_points_x_y.append(real_coor[:, :2])


def calibrate():

    H = get_homography(pic_points, real_points_x_y)
    intrinsics_param = get_intrinsics_param(H)
    extrinsics_param = get_extrinsics_param(H, intrinsics_param)
    k = get_distortion(intrinsics_param, extrinsics_param, pic_points, real_points_x_y)
    [new_intrinsics_param, new_k, new_extrinsics_param] = refinall_all_param(intrinsics_param,
                                                                             k, extrinsics_param, real_points,
                                                                             pic_points)

    print("intrinsics_parm:\t", new_intrinsics_param)
    print("distortionk:\t", new_k)
    print("extrinsics_parm:\t", new_extrinsics_param)

    print("intrinsics_parm:\t", np.linalg.norm(new_intrinsics_param - intrinsics_param))
    print("distortionk:\t", np.linalg.norm(new_k - k))
    print("extrinsics_parm:\t", np.linalg.norm(new_extrinsics_param - extrinsics_param))


if __name__ == "__main__":

    dir = './pic'
    row = 8
    column = 11
    unit_length = 0.02

    real_points = []
    real_points_x_y = []
    pic_points = []

    extract_point(dir, row, column, unit_length)
    calibrate()
