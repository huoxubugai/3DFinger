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
    outer_para_preprocess()
    cameras_coordinate_preprocess()


# 相机外参预处理
def outer_para_preprocess():
    for j in range(0, 6):
        a = random.uniform(-0.1, 0.1)
        b = random.uniform(-0.1, 0.1)
        c = random.uniform(-0.1, 0.1)
        d = random.uniform(-0.1, 0.1)
        e = random.uniform(-0.1, 0.1)
        f = random.uniform(-0.1, 0.1)
        g = random.uniform(-0.1, 0.1)
        h = random.uniform(-0.1, 0.1)
        i = random.uniform(-0.1, 0.1)
        t1 = random.uniform(-0.1, 0.1)
        t2 = random.uniform(-0.1, 0.1)
        t3 = random.uniform(-0.1, 0.1)
        random_outer_para_change = np.mat([[a, b, c, t1],
                                           [d, e, f, t2],
                                           [g, h, i, t3],
                                           [0, 0, 0, 0]])
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
    camera_plane_para = pfd.get_camera_plane(cameras_coordinate)
    tl.camera_plane_para = camera_plane_para
    # 获取A，E，F的映射点
    camera_a_point = tl.get_mapping_point_in_camera_plane(cameras_coordinate[0], camera_plane_para)
    camera_e_point = tl.get_mapping_point_in_camera_plane(cameras_coordinate[4], camera_plane_para)
    camera_f_point = tl.get_mapping_point_in_camera_plane(cameras_coordinate[5], camera_plane_para)
    # 六个相机归到一个平面之后的坐标：BCD不变，AEF映射到BCD平面
    cameras_coordinate_mapping = [camera_a_point, cameras_coordinate[1], cameras_coordinate[2],
                                  cameras_coordinate[3], camera_e_point, camera_f_point]
    cameras_coordinate_mapping = np.array(cameras_coordinate_mapping)
    tl.cameras_coordinate_mapping = cameras_coordinate_mapping


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


def recover_variables():
    pass


if __name__ == '__main__':
    dir_path = 'outer_files/LFMB_Visual_Hull_Meshes256'
    path_list = os.listdir(dir_path)
    total_loss = 0
    for i in range(0, 1):
        # 6个相机外参预处理：由于相机外参在不断改变，因此每次都需要重新计算投影矩阵、相机三维坐标、相机平面方程、相机映射坐标
        pre_process()

        for path in path_list:
            path_str = path.split(".")
            cur_file_path = dir_path + '/' + path_str[0]
            cur_loss = main_base_face.main(cur_file_path)
            print("目前损失为：", cur_loss)
            total_loss += cur_loss
            # todo 恢复所有变更了的全局变量
            recover_changed_variables()
        print("总损失为：", total_loss)

        # todo  目前需要复原全局变量，否则会一直累积
        # recover_cameras_related_variables()
