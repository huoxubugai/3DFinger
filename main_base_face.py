# -*- coding: utf-8 -*-
"""
@Time ： 2020/11/13 14:27
@Auth ： 零分
@File ：main_base_face.py
@IDE ：PyCharm
@github:https://github.com/huoxubugai/3DFinger
"""

from process import process_finger_data as pfd, faces_texture_mapping as ftm
from tool import tools as tl
import time

# todo 所有异常处理，包括文件读取异常，除零异常等等
'通过面进行纹理映射'
if __name__ == '__main__':
    start = time.time()
    file_path = 'outer_files/LFMB_Visual_Hull_Meshes256/0001_2_01'
    obj_suffix = '.obj'
    # 拿到mesh所有顶点数据
    data_points, face_start_index = pfd.read_mesh_points(file_path + obj_suffix)
    # 求出所有顶点对应的中心点O
    center_point = pfd.get_center_point(data_points)
    # 获取相机平面的参数ax+by+cz+d=0,直接使用计算好的数据
    camera_plane_para = tl.camera_plane_para
    # 获取中心点O的映射点
    center_point_mapping = tl.get_mapping_point_in_camera_plane(center_point, camera_plane_para)
    # 将mesh顶点数据中的所有顶点映射到相机平面
    data_points_mapping = pfd.get_data_points_mapping(data_points, camera_plane_para)
    # 数据预处理完毕，寻找每个点对应的相机；这里注意找到相机之后需要添加到源数据点上，而不是映射后的数据点
    camera_index_to_points = pfd.get_data_points_from_which_camera(center_point_mapping, data_points_mapping,
                                                                   tl.cameras_coordinate_mapping, data_points)

    # 纹理映射部分，这里和之前先后顺序不同，要从三角面片出发，得到每个面对应的相机，再将三角面片上的三个顶点投影到这个相机对应的bmp图片上，找到uv值
    faces_point = pfd.read_mesh_faces(file_path + obj_suffix, face_start_index)  # 读取obj中face的顶点数据
    faces_texture = ftm.mapping_faces_gray(data_points, camera_index_to_points, faces_point, file_path)  # 拿到所有面的纹理区域
    print("程序执行时间为:", time.time() - start, "秒")
