# -*- coding: utf-8 -*-
"""
@Time ： 2020/11/13 14:27
@Auth ： 零分
@File ：main.py
@IDE ：PyCharm
@github:https://github.com/huoxubugai/3DFinger
"""

from process import process_finger_data as pfd, points_texture_mapping as tm
import numpy as np
from tool import tools as tl
import os
import time

'通过mesh顶点进行纹理映射'
if __name__ == '__main__':
    # todo 遍历文件夹下的所有mesh 操作每一个
    file_path = 'outer_files/LFMB_Visual_Hull_Meshes256/0001_resize'
    obj_suffix = '.obj'
    uv_file_path = file_path + '.txt'
    # 先判断是否有生成的含像素uv的txt数据文件，如果有，直接读取txt，否则进行数据处理生成含uv的txt
    if os.path.exists(uv_file_path):
        # 读取uv文件
        uv_points = pfd.read_uv_points(uv_file_path)
        # tl.print_data_points(uv_points)
        start = time.time()
        points_gray = tm.mapping_points_gray(uv_points, file_path)  # 拿到所有对应点的灰度值（rgb）
        print(time.time() - start)
        # tl.print_data_points(points_gray)
        # 拿到灰度值list 写入到obj文件中
        tm.write_gray_to_obj(points_gray, file_path)
    else:
        # todo 文件读取异常处理
        # 拿到mesh所有顶点数据
        data_points = pfd.read_mesh_points(file_path + obj_suffix)  # 数据点的数据结构选择list而不是数组,方便后续改动
        print("原始数据点为：", data_points)
        # 求出所有顶点对应的中心点O
        center_point = pfd.get_center_point(data_points)
        print("中心点是：\n", center_point)
        # 求出六个相机在世界坐标系下的原点
        camera_points = pfd.get_all_camera_origin()
        print("六个相机的原点为：\n", camera_points)
        # 获取相机平面的参数ax+by+cz+d=0
        camera_plane_para = pfd.get_camera_plane(camera_points)
        print("六个相机的平面参数为：", camera_plane_para)
        # 获取O，A，E，F的映射点
        center_point_mapping = tl.get_mapping_point_in_camera_plane(center_point, camera_plane_para)
        print("映射后的中心点O坐标：\n", center_point_mapping)
        camera_a_point = tl.get_mapping_point_in_camera_plane(camera_points[0], camera_plane_para)
        camera_e_point = tl.get_mapping_point_in_camera_plane(camera_points[4], camera_plane_para)
        camera_f_point = tl.get_mapping_point_in_camera_plane(camera_points[5], camera_plane_para)
        # 六个相机归到一个平面之后的坐标：BCD不变，AEF映射到BCD平面
        camera_point_mapping = [camera_a_point, camera_points[1], camera_points[2],
                                camera_points[3], camera_e_point, camera_f_point]
        camera_point_mapping = np.array(camera_point_mapping)
        print("映射后的相机坐标：\n", camera_point_mapping)
        # 将mesh顶点数据中的所有顶点映射到相机平面
        data_points_mapping = pfd.get_data_points_mapping(data_points, camera_plane_para)
        print("映射后的所有数据点：\n", data_points_mapping)
        # ifd.print_data_points(data_points_mapping)
        # 数据预处理完毕，寻找每个点对应的相机
        # 这里注意找到相机之后需要添加到源数据点上，而不是映射后的数据点
        data_points_contain_camera = pfd.get_data_points_from_which_camera(center_point_mapping, data_points_mapping,
                                                                           camera_point_mapping, data_points)
        print("获取所有数据点以及其来源的相机索引\n")
        tl.print_data_points(data_points_contain_camera)
        # 得到每个点是由什么相机拍摄之后，进行纹理映射部分
        # 得到每个点对应二维图像上的u，v值
        uv_for_points = tm.get_uv_for_points(data_points_contain_camera)
        tl.print_data_points(uv_for_points)
        # 将这些数据写入文件  以后处理直接从文件中读取
        np.savetxt(file_path + ".txt", uv_for_points, fmt='%.7f')
        uv_points = pfd.read_uv_points(uv_file_path)
        tl.print_data_points(uv_points)
        points_gray = tm.mapping_points_gray(uv_points, file_path)
        tl.print_data_points(points_gray)
        # 拿到灰度值list 写入到obj文件中
        tm.write_gray_to_obj(points_gray, file_path)
