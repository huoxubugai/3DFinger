import os
import numpy as np
import math


# 根据obj文件获得mesh的顶点数据
def get_mesh_point(obj_file_path):
    with open(obj_file_path) as file:
        points = []
        while 1:
            line = file.readline()
            if not line:
                break
            strs = line.split(" ")
            if strs[0] == "v":
                points.append((float(strs[1]), float(strs[2]), float(strs[3])))
            else:
                break
            # if strs[0] == "vt":
            #     break
    # points原本为列表，需要转变为矩阵，方便处理
    # points = np.array(points)
    return points


# 获得mesh的中心点数据
def get_center_point(points):
    x_total = 0
    y_total = 0
    z_total = 0
    for p in points:
        x_total += p[0]
        y_total += p[1]
        z_total += p[2]
    size = len(points)
    center_point = [x_total / size, y_total / size, z_total / size]
    center_point = np.array(center_point)
    return center_point


# 根据相机外参获得相机在世界坐标系下的坐标
def get_single_camera_origin(m2):
    # 先对外参矩阵求逆
    m2 = m2.I
    m2 = np.array(m2)
    # 按公式可得坐标就是逆矩阵中每行的最后一个元素
    origin = m2[:3, 3]  # 取前三行第四个元素即可
    return origin


def get_all_camera_origin():
    # 6个相机的外参
    a = np.mat([[0.574322111, 0.771054881, 0.275006333, 0.93847817],
                [0.565423192, -0.130698104, -0.814379899, -0.36935905],
                [-0.591988790, 0.623211341, -0.511035123, 4.78810628],
                [0, 0, 0, 1]])
    b = np.mat([[0.456023570, 0.727006744, 0.513326112, 1.72205846],
                [-0.146061166, 0.630108915, -0.762645980, -0.30452329],
                [-0.877900131, 0.272807532, 0.393531969, 5.53092307],
                [0, 0, 0, 1]])
    c = np.mat([[0.609183831, 0.528225460, 0.591500569, 1.59956459],
                [-0.738350101, 0.649953779, 0.179997814, 0.5030131],
                [-0.289368602, -0.546386263, 0.785956655, 5.58635091],
                [0, 0, 0, 1]])
    d = np.mat([[0.771746127, 0.478767298, 0.418556793, 0.955855425],
                [-0.476877262, 0.000270229651, 0.878969854, 0.477556906],
                [0.420708915, -0.877941799, 0.228521787, 4.61760675],
                [0, 0, 0, 1]])
    e = np.mat([[0.788882832, 0.555210653, 0.263448302, 0.71648894],
                [0.159053746, -0.598545227, 0.785140445, 0.00777088],
                [0.593604063, -0.577481378, -0.560490387, 4.30437514],
                [0, 0, 0, 1]])
    f = np.mat([[0.712321206, 0.689000523, 0.133704068, 1.13938413],
                [0.694227260, -0.719684989, 0.0101009224, -0.28640104],
                [0.103184351, 0.0856259076, -0.990969825, 4.49819911],
                [0, 0, 0, 1]])

    camera_origins = [get_single_camera_origin(a), get_single_camera_origin(b), get_single_camera_origin(c),
                      get_single_camera_origin(d), get_single_camera_origin(e), get_single_camera_origin(f)]
    # 将list转为array
    camera_origins = np.array(camera_origins)
    return camera_origins


# 获取相机平面ax+by+cz+d=0
def get_camera_plane(camera_point):
    # B相机坐标
    x1 = camera_point[1][0]
    y1 = camera_point[1][1]
    z1 = camera_point[1][2]
    # C相机坐标
    x2 = camera_point[2][0]
    y2 = camera_point[2][1]
    z2 = camera_point[2][2]
    # D相机坐标
    x3 = camera_point[3][0]
    y3 = camera_point[3][1]
    z3 = camera_point[3][2]

    a = (y2 - y1) * (z3 - z1) - (y3 - y1) * (z2 - z1)
    b = (z2 - z1) * (x3 - x1) - (z3 - z1) * (x2 - x1)
    c = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
    d = -a * x1 - b * y1 - c * z1
    plane_para = [a, b, c, d]
    return plane_para


# 点到空间平面的映射点
def get_mapping_point_in_camera_plane(point, camera_plane_para):
    a = camera_plane_para[0]
    b = camera_plane_para[1]
    c = camera_plane_para[2]
    d = camera_plane_para[3]
    x = point[0]
    y = point[1]
    z = point[2]
    temp = a * a + b * b + c * c
    x_ = ((b * b + c * c) * x - a * (b * y + c * z + d)) / temp
    y_ = ((a * a + c * c) * y - b * (a * x + c * z + d)) / temp
    z_ = ((a * a + b * b) * z - c * (a * x + b * y + d)) / temp
    point_ = [x_, y_, z_]
    return point_


# 数据集所有点映射到平面
def get_data_points_mapping(data_points, camera_plane_para):
    for i in range(len(data_points)):
        cur_point_mapping = get_mapping_point_in_camera_plane(data_points[i], camera_plane_para)
        data_points[i] = cur_point_mapping
    return data_points


# 方法一 根据映射投影矢量之间夹角最小来判断mesh上所有顶点来自哪个摄像机拍摄（与哪个摄像机最近）
def get_data_points_from_which_camera(center_point, data_points, camera_points):
    # 设当前顶点为N，中心点为0，两个相邻的相机为X，Y。则判断分为两步
    # 第二步：根据O_N_向量与OA,OB...等向量夹角，找到夹角最小的相机即为所选择

    # 计算一下每个相机出现的次数，判断是否均衡
    camera_index_count = [0, 0, 0, 0, 0, 0]
    for i in range(len(data_points)):
        cur_target_camera_index = get_single_point_from_which_camera(center_point, data_points[i], camera_points)
        data_points[i].append(cur_target_camera_index)  # 将找到的相机添加在当前数据后面
        camera_index_count[cur_target_camera_index] += 1
    print("每个相机出现的次数为：", camera_index_count)  # 分别为38, 49, 51, 36, 40, 42
    return data_points


# 判断mesh上单一顶点来自哪个摄像机拍摄（与哪个摄像机最近）
# 根据ON向量与OA,OB,...向量夹角比较，夹角越小，余弦值越大，即为所需
def get_single_point_from_which_camera(center_point, cur_point, camera_points):
    cur_vector = calculate_vector(center_point, cur_point)
    max_vector_cosine = -2  # 初始化最大值为一个很小的值
    target_camera_index = 0  # 初始化A相机是所求的相机下标，0代表A，1代表B 以此类推
    for i in range(len(camera_points)):
        camera_vector = calculate_vector(center_point, camera_points[i])
        cur_cosine = calculate_cosine(cur_vector, camera_vector)
        if cur_cosine > max_vector_cosine:
            max_vector_cosine = cur_cosine
            target_camera_index = i
    return target_camera_index


# 方法二 根据叉乘（向量积）来判断点来自于哪个相机
def get_point_from_which_camera2(cur_point, center_point, camera_points):
    cur_vector = calculate_vector(center_point, cur_point)
    count = 0
    for i in range(len(camera_points)):
        camera_vector1 = calculate_vector(center_point, camera_points[i])
        if i != len(camera_points) - 1:
            camera_vector2 = calculate_vector(center_point, camera_points[i + 1])
        else:
            camera_vector2 = calculate_vector(center_point, camera_points[0])  # F相机和A相机的情况
        vector_product1 = calculate_vector_product(camera_vector1, cur_vector)
        vector_product2 = calculate_vector_product(camera_vector2, cur_vector)
        # 判断计算出的两个向量积的夹角
        if calculate_cosine(vector_product1, vector_product2) <= 0:
            count += 1  # 只是判断是否存在一个点的值小于0
    return count


# 根据两个点计算向量
def calculate_vector(from_point, to_point):
    vector = [to_point[0] - from_point[0], to_point[1] - from_point[1], to_point[2] - from_point[2]]
    return vector


# 计算两个向量的向量积
# AB=(x1,y1,z1)  CD=(x2,y2,z2) cross(AB,CD)=(y1*z2-y2z1,z1x2-z2x1,x1y2-x2y1)
def calculate_vector_product(vector1, vector2):
    vector_product = [vector1[1] * vector2[2] - vector1[2] * vector2[1],
                      vector1[2] * vector2[0] - vector1[0] * vector2[2],
                      vector1[0] * vector2[1] - vector1[1] * vector2[0]]
    return vector_product


# 计算两个向量的夹角是否大于90度 即夹角的余弦值是否小于0
# 公式为cos<a,b>=a.b/|a||b|. a.b=(x1x2+y1y2+z1z2) |a|=√(x1^2+y1^2+z1^2), |b|=√(x2^2+y2^2+z2^2).
def calculate_cosine(vector1, vector2):
    a = vector1[0] * vector2[0] + vector1[1] * vector2[1] + vector1[2] * vector2[2]
    b = math.sqrt(vector1[0] * vector1[0] + vector1[1] * vector1[1] + vector1[2] * vector1[2])
    c = math.sqrt(vector2[0] * vector2[0] + vector2[1] * vector2[1] + vector2[2] * vector2[2])
    res = a / (b * c)
    return res


def print_data_points(data_points):
    for li in data_points:
        print(li)
