# -*- coding: utf-8 -*-
"""
@Time ： 2020/12/17 10:33
@Auth ： 零分
@File ：main_base_optimization.py
@IDE ：PyCharm
@github:https://github.com/huoxubugai/3DFinger
"""
import main_base_face
import os
import numpy as np
import random
from tool import tools as tl
from process import process_finger_data as pfd


# 6个相机外参预处理：由于相机外参在不断改变，因此每次都需要重新计算投影矩阵、相机三维坐标、相机平面方程、相机映射坐标
def pre_process():
    outer_para_preprocess(0, 5)
    cameras_coordinate_preprocess()


# 相机外参预处理 选定变化的相机范围
def outer_para_preprocess(camera_index_start, camera_index_end):
    # 6个相机的外参，需要保存6分随机因子
    for j in range(camera_index_start, camera_index_end + 1):
        random_range = [-0.07, 0.07]
        a = random.uniform(random_range[0], random_range[1])
        b = random.uniform(random_range[0], random_range[1])
        c = random.uniform(random_range[0], random_range[1])
        d = random.uniform(random_range[0], random_range[1])
        e = random.uniform(random_range[0], random_range[1])
        f = random.uniform(random_range[0], random_range[1])
        g = random.uniform(random_range[0], random_range[1])
        h = random.uniform(random_range[0], random_range[1])
        i = random.uniform(random_range[0], random_range[1])
        t1 = random.uniform(random_range[0], random_range[1])
        t2 = random.uniform(random_range[0], random_range[1])
        t3 = random.uniform(random_range[0], random_range[1])
        random_outer_para_change = np.mat([[a, b, c, t1],
                                           [d, e, f, t2],
                                           [g, h, i, t3],
                                           [0, 0, 0, 0]])
        # 将随机因子写回全局变量
        tl.cur_random_cameras_outer_para[j] = random_outer_para_change
        tl.cameras_outer_para[j] += random_outer_para_change
        cur_projection_mat = tl.cameras_inner_para[j] * tl.cameras_outer_para[j]
        tl.all_camera_projection_mat[j] = cur_projection_mat


def cameras_coordinate_preprocess():
    cameras_coordinate = [pfd.get_single_camera_origin(tl.cameras_outer_para[0]),
                          pfd.get_single_camera_origin(tl.cameras_outer_para[1]),
                          pfd.get_single_camera_origin(tl.cameras_outer_para[2]),
                          pfd.get_single_camera_origin(tl.cameras_outer_para[3]),
                          pfd.get_single_camera_origin(tl.cameras_outer_para[4]),
                          pfd.get_single_camera_origin(tl.cameras_outer_para[5])]
    # 将list转为array
    cameras_coordinate = np.array(cameras_coordinate)
    camera_plane_para_bcd = pfd.get_camera_plane_bcd(cameras_coordinate)
    camera_plane_para_abf = pfd.get_camera_plane_abf(cameras_coordinate)
    tl.camera_plane_para = camera_plane_para_abf
    # # 获取A，B、C、D、E，F在bcd平面的映射点
    # camera_a_point = tl.get_mapping_point_in_camera_plane(cameras_coordinate[0], camera_plane_para_bcd)
    # camera_b_point = tl.get_mapping_point_in_camera_plane(cameras_coordinate[1], camera_plane_para_bcd)
    # camera_c_point = tl.get_mapping_point_in_camera_plane(cameras_coordinate[2], camera_plane_para_bcd)
    # camera_d_point = tl.get_mapping_point_in_camera_plane(cameras_coordinate[3], camera_plane_para_bcd)
    # camera_e_point = tl.get_mapping_point_in_camera_plane(cameras_coordinate[4], camera_plane_para_bcd)
    # camera_f_point = tl.get_mapping_point_in_camera_plane(cameras_coordinate[5], camera_plane_para_bcd)
    # 获取A，B、C、D、E，F在abf平面的映射点
    camera_c_point = tl.get_mapping_point_in_camera_plane(cameras_coordinate[2], camera_plane_para_abf)
    camera_d_point = tl.get_mapping_point_in_camera_plane(cameras_coordinate[3], camera_plane_para_abf)
    camera_e_point = tl.get_mapping_point_in_camera_plane(cameras_coordinate[4], camera_plane_para_abf)
    # 六个相机归到一个平面之后的坐标：
    cameras_coordinate_mapping = [cameras_coordinate[0], cameras_coordinate[1], camera_c_point,
                                  camera_d_point, camera_e_point, cameras_coordinate[5]]
    cameras_coordinate_mapping = np.array(cameras_coordinate_mapping)
    tl.cameras_coordinate_mapping = cameras_coordinate_mapping


def calculate_gradient():
    pass


# 每次完整的操作之后 都要恢复被改变的全局变量
def recover_changed_variables():
    tl.bmp_pixel = [[], [], [], [], [], []]
    tl.map_vertex_to_texture = dict()
    tl.map_vertex_to_vt_index = dict()
    tl.bmp_crop_ranges = [[10000, 10000, -100, -100], [10000, 10000, -100, -100],
                          [10000, 10000, -100, -100], [10000, 10000, -100, -100],
                          [10000, 10000, -100, -100], [10000, 10000, -100, -100]]
    tl.crops_width_and_height = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    tl.crops_v_scale_in_png = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
    tl.uv_map_size = [0, 0]
    tl.vertex_in_faces_belong_camera = []


def recover_cameras_related_variables():
    # 应该只需要改这一个变量(会累积)
    tl.cameras_outer_para = [
        np.mat([[0.574322111, 0.771054881, 0.275006333, 0.93847817],
                [0.565423192, -0.130698104, -0.814379899, -0.36935905],
                [-0.591988790, 0.623211341, -0.511035123, 4.78810628],
                [0, 0, 0, 1]]),
        np.mat([[0.456023570, 0.727006744, 0.513326112, 1.72205846],
                [-0.146061166, 0.630108915, -0.762645980, -0.30452329],
                [-0.877900131, 0.272807532, 0.393531969, 5.53092307],
                [0, 0, 0, 1]]),
        np.mat([[0.609183831, 0.528225460, 0.591500569, 1.59956459],
                [-0.738350101, 0.649953779, 0.179997814, 0.5030131],
                [-0.289368602, -0.546386263, 0.785956655, 5.58635091],
                [0, 0, 0, 1]]),
        np.mat([[0.771746127, 0.478767298, 0.418556793, 0.955855425],
                [-0.476877262, 0.000270229651, 0.878969854, 0.477556906],
                [0.420708915, -0.877941799, 0.228521787, 4.61760675],
                [0, 0, 0, 1]]),
        np.mat([[0.788882832, 0.555210653, 0.263448302, 0.71648894],
                [0.159053746, -0.598545227, 0.785140445, 0.00777088],
                [0.593604063, -0.577481378, -0.560490387, 4.30437514],
                [0, 0, 0, 1]]),
        np.mat([[0.712321206, 0.689000523, 0.133704068, 1.13938413],
                [0.694227260, -0.719684989, 0.0101009224, -0.28640104],
                [0.103184351, 0.0856259076, -0.990969825, 4.49819911],
                [0, 0, 0, 1]])
    ]
    tl.cur_random_cameras_outer_para = [
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
        [[0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0]],
    ]
    tl.all_camera_projection_mat = [
        [[1.39434783e+02, 1.18422163e+03, -9.32437833e+01, 4.27466162e+03],
         [3.39496212e+02, 9.22510264e+01, -9.67653298e+02, 1.32319794e+03],
         [-5.91988790e-01, 6.23211341e-01, -5.11035123e-01, 4.78810628e+00]],
        [[-7.85090956e+01, 8.61230229e+02, 7.26596598e+02, 4.92106359e+03],
         [-5.02774485e+02, 7.19172239e+02, -5.71889964e+02, 1.98846331e+03],
         [-8.77900131e-01, 2.72807532e-01, 3.93531969e-01, 5.53092307e+00]],
        [[4.12009678e+02, 1.76193887e+02, 1.05384338e+03, 4.97065152e+03],
         [-8.45497311e+02, 3.82381880e+02, 5.29296949e+02, 3.01051417e+03],
         [-2.89368602e-01, -5.46386263e-01, 7.85956655e-01, 5.58635091e+00]],
        [[1.03315200e+03, -1.48038125e+02, 5.60572927e+02, 4.11740670e+03],
         [-2.82474656e+02, -3.66226258e+02, 9.39743146e+02, 2.38630951e+03],
         [4.20708915e-01, -8.77941799e-01, 2.28521787e-01, 4.61760675e+00]],
        [[1.18728070e+03, 1.08759358e+02, -1.57607533e+02, 3.82810628e+03],
         [4.19718174e+02, -8.31607535e+02, 4.95766722e+02, 1.95088770e+03],
         [5.93604063e-01, -5.77481378e-01, -5.60490387e-01, 4.30437514e+00]],
        [[7.46729038e+02, 7.13841054e+02, -4.61241373e+02, 3.77373081e+03],
         [7.08169289e+02, -6.57709441e+02, -3.83547441e+02, 1.50980066e+03],
         [1.03184351e-01, 8.56259076e-02, -9.90969825e-01, 4.49819911e+00]]
    ]
    tl.camera_plane_para = [19.467678495159983, 18.098947303577706, 10.253452426300939, 1.884526845005233]
    tl.cameras_coordinate_mapping = [[2.45592658, -3.80092362, 1.86249467],
                                     [4.02581981, -2.56894275, -3.29281609],
                                     [1.01348544, 1.88043939, -5.4273143],
                                     [-2.45261002, 3.5962286, -1.87506165],
                                     [-3.16297766, 2.05403639, 2.19588564],
                                     [-1.08130466, -1.38038999, 4.30582486]]


'注意事项：1、用的是什么相机平面 2、优化的是什么相机范围，默认0-5全相机，范围不同边界点也不一样 3、随机因子范围'
if __name__ == '__main__':
    dir_path = 'outer_files/LFMB_Visual_Hull_Meshes256'
    path_list = os.listdir(dir_path)
    cur_total_loss = 0
    min_total_loss = 9999999  # 设定最大值为一个较大的数
    for i in range(0, 200):
        # 6个相机外参预处理：由于相机外参在不断改变，因此每次都需要重新计算投影矩阵、相机三维坐标、相机平面方程、相机映射坐标
        pre_process()
        for path in path_list:
            path_str = path.split(".")
            cur_file_path = dir_path + '/' + path_str[0]
            cur_loss = main_base_face.main(cur_file_path)
            # print("目前损失为：", cur_loss)
            cur_total_loss += cur_loss
            # todo 恢复所有变更了的全局变量
            recover_changed_variables()
        print("第", i, "次的总损失为：", cur_total_loss)
        if cur_total_loss < min_total_loss:
            min_total_loss = cur_total_loss
            tl.optimal_random_cameras_outer_para = tl.cur_random_cameras_outer_para  # 将当前的随机因子写入最佳随机因子

        print("当前最小总损失为：", min_total_loss)
        # 损失重置为0
        cur_total_loss = 0
        # todo  目前需要复原全局变量，否则会一直累积
        recover_cameras_related_variables()
        if i % 10 == 0:
            print("当前最优随机因子为:", tl.optimal_random_cameras_outer_para)
    print("最优随机因子为：", tl.optimal_random_cameras_outer_para)
    # todo  损失比较，随机因子保存
