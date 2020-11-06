import numpy as np
from tool import tools as tl
import cv2


# 面的纹理映射
def mapping_faces_gray(uv_points, faces_point, file_path):
    # 得出每个三角面片都属于哪一个相机，两个点及以上属于一个相机，则该面属于该相机
    faces_point_contain_camera = get_faces_belong_which_camera(uv_points, faces_point)
    # todo 考虑是否需要将这个写入本地文件

    pass


# 获得三角面片属于什么相机
def get_faces_belong_which_camera(uv_points, faces_point):
    faces_point_contain_camera = []
    for face in faces_point:
        face_with_camera = []
        for v in face:
            camera_index = uv_points[v - 1][0]
            face_with_camera.append(camera_index)
        max_count_camera = max(face_with_camera, key=face_with_camera.count)  # 得出出现最多次数的相机索引
        face.append(max_count_camera)
        faces_point_contain_camera.append(face)
    return faces_point_contain_camera


def write_gray_to_obj(points_gray, obj_file_path):
    pass
