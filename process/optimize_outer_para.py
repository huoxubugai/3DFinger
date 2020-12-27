# -*- coding: utf-8 -*-
"""
@Time ： 2020/12/17 11:17
@Auth ： 零分
@File ：optimize_outer_para.py
@IDE ：PyCharm
@github:https://github.com/huoxubugai/3DFinger
"""
from process import points_texture_mapping as ptm
from tool import tools as tl


# 返回数据结构格式为[[1,0,1]...[9,2,3]...] 意为顶点1所在某一三角面片属于A相机(0)，自身属于B相机(1),
# 顶点9所在某一三角面片属于C相机(2)，自身属于D相机(3)
def get_all_points_in_edge(faces_points, vertex_in_faces_belong_camera):
    points_in_edge = []
    for i in range(0, len(faces_points)):
        cur_vertex_in_face_camera = vertex_in_faces_belong_camera[i]
        cur_face = faces_points[i]
        a = cur_vertex_in_face_camera[0]
        b = cur_vertex_in_face_camera[1]
        c = cur_vertex_in_face_camera[2]
        if (a == b and b == c) or (a != b and b != c and c != a):  # 如果a b c全相等或者全不相等则该面不属于边界
            continue
        else:
            if a == b:
                cur_point_in_edge = [cur_face[2], a, c]
            elif a == c:
                cur_point_in_edge = [cur_face[1], a, b]
            else:
                cur_point_in_edge = [cur_face[0], b, a]
            if cur_point_in_edge not in points_in_edge:  # 对列表去重
                points_in_edge.append(cur_point_in_edge)
    return points_in_edge


# 获取指定两个相机之间的边界点
def get_points_in_edge_between_special_camera(faces_points, vertex_in_faces_belong_camera, camera_index1,
                                              camera_index2):
    points_in_edge = []
    for i in range(0, len(faces_points)):
        cur_vertex_in_face_camera = vertex_in_faces_belong_camera[i]
        cur_face = faces_points[i]
        a = cur_vertex_in_face_camera[0]
        b = cur_vertex_in_face_camera[1]
        c = cur_vertex_in_face_camera[2]
        if (a == b and b == c) or (a != b and b != c and c != a):  # 如果a b c全相等或者全不相等则该面不属于边界
            continue
        elif (a == camera_index1 or a == camera_index2) and (b == camera_index1 or b == camera_index2) \
                and (c == camera_index1 or c == camera_index2):
            if a == b:
                cur_point_in_edge = [cur_face[2], a, c]
            elif a == c:
                cur_point_in_edge = [cur_face[1], a, b]
            else:
                cur_point_in_edge = [cur_face[0], b, a]
            if cur_point_in_edge not in points_in_edge:  # 对列表去重
                points_in_edge.append(cur_point_in_edge)
    return points_in_edge


def get_points_gray_in_2_pic(points_in_edge, data_points, file_path):
    edge_points_grays = []
    for edge_point in points_in_edge:
        point_grays = get_point_grays(edge_point, data_points, file_path)
        edge_points_grays.append(point_grays)
    return edge_points_grays


def get_point_grays(edge_point, data_points, file_path):
    point_index = edge_point[0]
    point_in_face_camera_index = edge_point[1]
    point_belong_camera_index = edge_point[2]
    vertex_data = data_points[point_index - 1]  # 这里应该减一
    gray1 = get_gray_for_point(vertex_data, point_in_face_camera_index, file_path)
    gray2 = get_gray_for_point(vertex_data, point_belong_camera_index, file_path)
    return [point_index, gray1, gray2]


def get_gray_for_point(vertex_data, camera_index, file_path):
    path_str = file_path.split("/")
    camera_name = tl.camera_index_to_name[camera_index]
    pic_path_prefix = 'outer_files/images/' + path_str[2]  # todo 注意这里的索引会随着文件路径改变而改变
    pic_file_path = pic_path_prefix + '_' + camera_name + '.bmp'  # 拼接文件名
    cur_uv = ptm.get_texture_for_single_point(vertex_data, camera_index)
    # 归到正确的范围内
    if cur_uv[0] > tl.cur_pic_size[0]:
        cur_uv[0] = tl.cur_pic_size[0]
    if cur_uv[1] > tl.cur_pic_size[1]:
        cur_uv[1] = tl.cur_pic_size
    if cur_uv[0] <= 0:
        cur_uv[0] = 1
    if cur_uv[1] <= 0:
        cur_uv[1] = 1
    rgb_gray = ptm.get_pic_gray(pic_file_path, camera_index, cur_uv[0], cur_uv[1])
    gray = (rgb_gray[0] * 299 + rgb_gray[1] * 587 + rgb_gray[2] * 114) / 1000  # 将r,g，b转换为灰度值
    gray = round(gray)
    return gray


def calculate_loss(edge_points_grays):
    loss = 0
    size = len(edge_points_grays)
    for grays in edge_points_grays:
        gray_difference = grays[1] - grays[2]
        loss += (gray_difference * gray_difference)
    # todo  考虑是否除以size
    return loss / size
