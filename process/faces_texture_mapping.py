import numpy as np
from tool import tools as tl
import cv2
import matplotlib.pyplot as plt


# 面的纹理映射
def mapping_faces_gray(data_points_contain_camera, faces_point, file_path):
    # 得出每个三角面片都属于哪一个相机，两个点及以上属于一个相机，则该面属于该相机
    faces_point_contain_camera = get_faces_belong_which_camera(data_points_contain_camera, faces_point)
    # todo 考虑是否需要将这个写入本地文件
    # 拿到三角面片对应的相机后，对该三角面片做相应图片的映射，三维——》二维
    for face in faces_point_contain_camera:
        get_texture_from_bmp(face, data_points_contain_camera)  # 会将所有三角面片对应点的纹理全放进哈希表中,同时对面片按相机分类
    # todo 按不同相机遍历face，根据face的顶点，从哈希表中分别将不同的相机的纹理值取出，记住其中的最大最小值，对对应bmp做crop放进png中
    # 根据全局变量bmp_crop_ranges，去对bmp图片做crop，然后放入uv map png图片中
    uv_map_png = crop_bmp_to_png(file_path)
    # 对每个面进行遍历，获取面上的点再uvmap_png中的对应uv值，然后按预期格式会写到obj文件中
    uv_val_in_obj, vt_list = get_png_uv_from_crops(faces_point)
    write_uv_to_obj(uv_val_in_obj, vt_list, file_path)


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
    # 将face根据不同的相机放进全局变量
    tl.faces_belong_camera[camera_index].append(face)
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
        # todo 为什么u会出现负数
        # if u < 0:
        #     print(u)
        u = round(u)
        v = round(v)
        # todo  uv 取整时不应该超过uv的应有范围，后续还是应该采用精度更高的做法，另外uv和像素矩阵的对应关系也应该确定是否是v-1.u-1
        # 由于(u,v)只代表像素的列数与行数,而四舍五入存在误差，为了不超过uv的范围，将它强行归到1-1280范围，1-800范围
        if u <= 0:
            u = 1
        if v <= 0:
            v = 1
        # 根据相机索引和像素点下标拼接key值，然后将uv放到哈希表中
        tl.map_vertex_to_texture[key] = [u, v]
        # 同时更新全局变量中的uv crop范围
        tl.bmp_crop_ranges[camera_index][0] = min(u, tl.bmp_crop_ranges[camera_index][0])
        tl.bmp_crop_ranges[camera_index][1] = min(v, tl.bmp_crop_ranges[camera_index][1])
        tl.bmp_crop_ranges[camera_index][2] = max(u, tl.bmp_crop_ranges[camera_index][2])
        tl.bmp_crop_ranges[camera_index][3] = max(v, tl.bmp_crop_ranges[camera_index][3])


def crop_bmp_to_png(file_path):
    # 初始化png大小，全0,png组成为A B C D E F 竖排摆放
    # 计算各相机crop出的宽度和高度
    calculate_crop_width_and_height()
    png_width = max(tl.crops_width_and_height[0][0], tl.crops_width_and_height[1][0], tl.crops_width_and_height[2][0],
                    tl.crops_width_and_height[3][0], tl.crops_width_and_height[4][0], tl.crops_width_and_height[5][0])
    png_height = tl.crops_width_and_height[0][1] + tl.crops_width_and_height[1][1] + tl.crops_width_and_height[2][1] + \
                 tl.crops_width_and_height[3][1] + tl.crops_width_and_height[4][1] + tl.crops_width_and_height[5][1]
    uv_map_png = np.zeros((png_height, png_width), dtype=np.uint8)
    # 放入全局变量
    tl.uv_map_size[:] = png_width, png_height
    for i in range(0, 6):
        cur_crop_range = tl.bmp_crop_ranges[i]
        cur_crop_bmp = crop_bmp(cur_crop_range, i, file_path)
        # 将crop出的图放入png中
        put_crop_into_png(cur_crop_bmp, uv_map_png, i)
    # 将png写入本地
    cv2.imwrite(file_path + '.png', uv_map_png)
    return uv_map_png


# 计算crop出的图片宽度和高度
def calculate_crop_width_and_height():
    tl.crops_width_and_height[0] = [tl.bmp_crop_ranges[0][2] - tl.bmp_crop_ranges[0][0],
                                    tl.bmp_crop_ranges[0][3] - tl.bmp_crop_ranges[0][1]]
    tl.crops_width_and_height[1] = [tl.bmp_crop_ranges[1][2] - tl.bmp_crop_ranges[1][0],
                                    tl.bmp_crop_ranges[1][3] - tl.bmp_crop_ranges[1][1]]
    tl.crops_width_and_height[2] = [tl.bmp_crop_ranges[2][2] - tl.bmp_crop_ranges[2][0],
                                    tl.bmp_crop_ranges[2][3] - tl.bmp_crop_ranges[2][1]]
    tl.crops_width_and_height[3] = [tl.bmp_crop_ranges[3][2] - tl.bmp_crop_ranges[3][0],
                                    tl.bmp_crop_ranges[3][3] - tl.bmp_crop_ranges[3][1]]
    tl.crops_width_and_height[4] = [tl.bmp_crop_ranges[4][2] - tl.bmp_crop_ranges[4][0],
                                    tl.bmp_crop_ranges[4][3] - tl.bmp_crop_ranges[4][1]]
    tl.crops_width_and_height[5] = [tl.bmp_crop_ranges[5][2] - tl.bmp_crop_ranges[5][0],
                                    tl.bmp_crop_ranges[5][3] - tl.bmp_crop_ranges[5][1]]


def crop_bmp(crop_range, camera_index, file_path):
    # 拼接bmp文件路径，拿到bmp，对其crop
    path_str = file_path.split("/")
    camera_name = tl.camera_index_to_name[camera_index]
    pic_path_prefix = 'outer_files/images/' + path_str[2]  # todo 注意这里的索引会随着文件路径改变而改变
    pic_file_path = pic_path_prefix + '_' + camera_name + '.bmp'  # 拼接文件名
    cur_img = cv2.imread(pic_file_path, cv2.IMREAD_GRAYSCALE)
    # 根据crop range进行crop
    crop_img = cur_img[tl.bmp_crop_ranges[camera_index][1]:tl.bmp_crop_ranges[camera_index][3],
               tl.bmp_crop_ranges[camera_index][0]:tl.bmp_crop_ranges[camera_index][2]]
    # plt.imshow(crop_img, cmap="gray")
    # plt.show()
    # 将crop放入png
    return crop_img


def put_crop_into_png(crop_pic, uv_map_png, camera_index):
    v_start = 0
    crop_height = crop_pic.shape[0]
    crop_wight = crop_pic.shape[1]
    i = 0
    while i < camera_index:
        v_start += tl.crops_width_and_height[i][1]  # 累积前面的高度
        i += 1
    uv_map_png[v_start:v_start + crop_height, 0:crop_wight] = crop_pic
    plt.imshow(uv_map_png, cmap="gray")
    plt.show()


# 获得obj中所需要的信息
def get_png_uv_from_crops(faces_point):
    vt_list = []  # 每一行放在obj文件中f i/_ j/_ k/_
    vt_uv_val = []  # 存放uv具体信息  u,v:0->1
    i = 1  # vt_index 按照obj规定 ，从1开始
    for face in faces_point:
        camera_index = face[3]
        vt_in_face = []
        for vertex in face[0:3]:
            key = str(camera_index) + "_" + str(vertex)
            # 先判断key是否存在于全局哈希表map_vertex_to_vt_index中，若存在，不用后续操作，直接取出
            if key not in tl.map_vertex_to_vt_index.keys():
                cur_texture = tl.map_vertex_to_texture[key]
                cur_uv_in_png = get_uv_from_png(cur_texture, camera_index)
                vt_uv_val.append(cur_uv_in_png)
                vt_in_face.append(i)
                # 将key和值i放入全局哈希表map_vertex_to_vt_index
                tl.map_vertex_to_vt_index[key] = i  # todo 使用i作为index时记得减一
                i += 1
            else:
                vt_in_face.append(tl.map_vertex_to_vt_index[key])
        vt_list.append(vt_in_face)
    tl.print_data_points(vt_uv_val)
    tl.print_data_points(vt_list)
    return vt_uv_val, vt_list


# 根据像素信息获取png中对应的uv uv范围为0-1
def get_uv_from_png(cur_texture, camera_index):
    png_u = (cur_texture[0] - tl.bmp_crop_ranges[camera_index][0]) / tl.uv_map_size[0]  # 这里uv还需要考虑crop前后的坐标变化
    cur_height, i = 0, 0
    while i < camera_index:
        cur_height += tl.crops_width_and_height[i][1]  # 累积上面的高度
        i += 1
    cur_height += (cur_texture[1] - tl.bmp_crop_ranges[camera_index][1])  # 再加上自身的高度
    png_v = cur_height / tl.uv_map_size[1]
    return [png_u, png_v]


def write_uv_to_obj(uv_val_in_obj, vt_list, file_path):
    lines = []
    with open(file_path + '.obj', 'r') as f:
        # 先添加首部的顶点数据
        for line in f:
            # line = line[0:-1] + " " + str(gray) + '\n'
            # line = line[0:-1] +
            # print(line)
            if line[0] == 'v':
                lines.append(line)
                continue
            else:
                break
        # 在中部放入计算出来的uv信息
        for uv_val in uv_val_in_obj:
            cur_str = 'vt' + " " + str(uv_val[0]) + " " + str(uv_val[1]) + '\n'
            lines.append(cur_str)
        # 在底部更新三角面片数据
        for line, vt_index in zip(f, vt_list):
            face = line.split(" ")  # 先将字符串按空格切分成数组,取出末尾换行符，再进行拼接
            cur_str = 'f' + " " + face[1] + "/" + str(vt_index[0]) + \
                      " " + face[2] + "/" + str(vt_index[1]) + " " + \
                      face[3].replace('\n', '') + "/" + str(vt_index[2]) + '\n'
            print(line)
            lines.append(cur_str)
    with open(file_path + '_new.obj', 'w+') as f_new:
        f_new.writelines(lines)


def write_gray_to_obj(points_gray, obj_file_path):
    pass
