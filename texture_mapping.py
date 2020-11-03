import numpy as np
import process_finger_data as pfd
import tools as tl
import read_24bit_bmp as bmp
import os
import cv2


# 根据处理好的数据和相机的内外参进行处理，后续可换成文件名形式进行读取数据
# 因为已经事先根据内外参得到了相机的投影矩阵，因此直接用投影矩阵计算
# 获取三维点在二维图像上的位置（u,v）
def get_uv_for_points(data_points_contain_camera):
    for i in range(len(data_points_contain_camera)):
        point_data_contain_uv = get_uv_for_single_point(data_points_contain_camera[i])
        data_points_contain_camera[i] = point_data_contain_uv
    return data_points_contain_camera


def get_uv_for_single_point(point_data):
    # 获取相机下标索引,通过索引下标获得对应的投影矩阵
    camera_index = point_data[3]
    camera_projection_mat = tl.all_camera_projection_mat[camera_index]
    camera_projection_mat = np.mat(camera_projection_mat)
    # 根据公式，将点的x,y,z坐标变为4*1矩阵形式，最后补1
    point_mat = np.mat([[point_data[0]],
                        [point_data[1]],
                        [point_data[2]],
                        [1]])
    # 将3*4投影矩阵与4*1点的矩阵相乘，得到一个3*1的结果
    res = camera_projection_mat * point_mat
    u = res[0, 0] / res[2, 0]
    v = res[1, 0] / res[2, 0]
    # uv 取整
    u = round(u)
    v = round(v)
    point_data.append(u)
    point_data.append(v)
    return point_data


# 获取所有数据点的灰度值
def mapping_points_gray(uv_points, file_path):
    path_str = file_path.split("/")
    picture_path_prefix = 'images/' + path_str[1]
    points_gray = []
    for i in range(len(uv_points)):
        cur_gray = mapping_single_point_gray(uv_points[i], picture_path_prefix)
        points_gray.append(cur_gray)
    return points_gray


# 获取单个点的灰度值
def mapping_single_point_gray(point, pic_path_prefix):
    camera_index = point[0]
    camera_index = round(camera_index)
    camera_name = tl.camera_index_to_name[camera_index]
    pic_file_path = pic_path_prefix + '_' + camera_name + '.bmp'  # 拼接文件名
    u = round(point[1])  # todo 后续做插值而不是取整
    v = round(point[2])
    # 打开图片，根据uv获取灰度值
    gray = get_pic_gray(pic_file_path, u, v)
    # point.append(gray)  # todo 这里的point应该换成原始obj数据
    return gray


# 根据图片路径和像素u,v获取像素点的灰度值
def get_pic_gray(pic_file_path, u, v):
    # if os.path.exists(pic_file_path + '.txt'):
    #     # todo 1、写入的txt文件为什么会后面缺失0？  2、读取bmp的灰度txt文件到对应的list中，预加载以节省时间
    #     cur_img = pfd.read_uv_points(pic_file_path + '.txt')
    # else:
    # cur_img = bmp.read_rows(pic_file_path) # 用自己写的函数去读取
    cur_img = cv2.imread(pic_file_path, cv2.IMREAD_GRAYSCALE)  # 用opencv去读取bmp 直接拿到灰度值
    # np.savetxt(pic_file_path + '.txt', cur_img, fmt='%d')
    gray = cur_img[v-1][u-1]  # todo 注意这里u，v和像素矩阵的索引是要反过来的，在图像坐标系中，u为横坐标，v为纵坐标，这里是v-1,u-1还是v u？
    # 出现了错误：IndexError: index 1280 is out of bounds for axis 0 with size 1280,说明这里需要减一
    return gray
