import numpy as np
from tool import tools as tl
import cv2


# 面的纹理映射
def mapping_faces_gray(data_points_contain_camera, faces_point, file_path):
    # 得出每个三角面片都属于哪一个相机，两个点及以上属于一个相机，则该面属于该相机
    faces_point_contain_camera = get_faces_belong_which_camera(data_points_contain_camera, faces_point)
    # todo 考虑是否需要将这个写入本地文件
    # 拿到三角面片对应的相机后，对该三角面片做相应图片的映射，三维——》二维
    for face in faces_point_contain_camera:
        get_texture_from_bmp(face, data_points_contain_camera)  # 会将所有三角面片对应点的纹理全放进哈希表中
    # todo 遍历face，根据face的顶点，从哈希表中分别将不同的相机的纹理值取出，记住其中的最大最小值，对对应bmp做crop放进png中
    pass


# 获得三角面片属于什么相机
def get_faces_belong_which_camera(data_points_contain_camera, faces_point):
    faces_point_contain_camera = []
    for face in faces_point:
        face_with_camera = []
        for v in face:
            camera_index = data_points_contain_camera[v - 1][3]
            face_with_camera.append(camera_index)
        max_count_camera = max(face_with_camera, key=face_with_camera.count)  # 得出出现最多次数的相机索引
        face.append(max_count_camera)
        faces_point_contain_camera.append(face)
    return faces_point_contain_camera


def get_texture_from_bmp(face, data_points_contain_camera):
    # 用face里面存储的点的索引，去data_points_contain_camera里拿到对应的数据点
    camera_index = face[3]
    for vertex_index in face[0:3]:  # 注意这里只有前三个才是顶点索引
        vertex_data = data_points_contain_camera[vertex_index - 1]
        get_texture_for_vertex(vertex_data, camera_index, vertex_index)


# 这个方法和之前求纹理的方法get_texture_for_single_point类似，只是输入输出参数不同
def get_texture_for_vertex(vertex_data, camera_index, vertex_index):
    key = str(camera_index) + "_" + str(vertex_index)
    # 判断哈希表中是否已经存在该数据，避免重复计算
    if key not in tl.map_vertex_to_texture.keys():
        camera_projection_mat = tl.all_camera_projection_mat[camera_index]
        camera_projection_mat = np.mat(camera_projection_mat)
        # 根据公式，将点的x,y,z坐标变为4*1矩阵形式，最后补1
        point_mat = np.mat([[vertex_data[0]],
                            [vertex_data[1]],
                            [vertex_data[2]],
                            [1]])
        # 将3*4投影矩阵与4*1点的矩阵相乘，得到一个3*1的结果
        res = camera_projection_mat * point_mat
        u = res[0, 0] / res[2, 0]
        v = res[1, 0] / res[2, 0]
        # uv 取整
        u = round(u)
        v = round(v)
        # 根据相机索引和像素点下标拼接key值，然后将uv放到哈希表中
        tl.map_vertex_to_texture[key] = [u, v]


def write_gray_to_obj(points_gray, obj_file_path):
    pass
