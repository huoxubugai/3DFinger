import unittest
from process.process_finger_data import *
import matplotlib.pyplot as plt
from tool import tools as tl
from tool import read_24bit_bmp as rbm
import cv2
import time


class Test(unittest.TestCase):

    # 显示相机平面位置
    def test_show_camera_plane(self):
        camera_origins = [[2.50436065, -3.75589484, 1.88800446],
                          [4.02581981, -2.56894275, -3.29281609],
                          [1.01348544, 1.88043939, -5.4273143],
                          [-2.45261002, 3.5962286, -1.87506165],
                          [-3.12155638, 2.09254542, 2.21770186],
                          [-1.07692383, -1.37631717, 4.3081322],
                          [-1.14422972e-01, - 9.73947995e-02, 2.05371429e-01],
                          [-3.02246150e-01, 6.58760412e-02, 2.73782622e-01],
                          [-4.98036103e-01, 2.93154709e-01, 2.44336074e-01],
                          [3.18449475e-01, - 1.73387355e-01, - 4.82361449e-01],
                          [1.11402481e+00, - 7.18796123e-01, - 1.03014576e+00],
                          [1.49183233e-01, - 4.59760317e-01, 3.44508327e-01]
                          # [0.3165423775809235, -0.10786089526290894, -0.5944049978975746],
                          # [-1.261181, -2.678640, -2.523260]
                          ]
        camera_origins = np.array(camera_origins)
        x = camera_origins[:, 0]
        y = camera_origins[:, 1]
        z = camera_origins[:, 2]

        ax = plt.subplot(111, projection='3d')
        ax.scatter(x[:], y[:], z[:], c='r')
        ax.set_zlabel('Z')  # 坐标轴
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
        plt.show()

    # 测试计算向量
    def test_calculate_vector(self):
        point1 = [1, 2, 3]
        point2 = [3, 2, 1]
        vector = calculate_vector(point1, point2)
        self.assertEqual(vector, [2, 0, -2])

    # 测试计算向量积函数
    def test_calculate_vector_product(self):
        vector1 = [1, 2, 3]
        vector2 = [3, 2, 1]
        vector_product = tl.calculate_vector_product(vector1, vector2)
        self.assertEqual(vector_product, [-4, 8, -4])

    # 测试计算向量夹角余弦值函数
    def test_calculate_cosine(self):
        vector1 = [1, 0, 0]
        vector2 = [0, 0, 1]
        res = tl.calculate_cosine(vector1, vector2)
        print(tl.calculate_cosine([1, 2, 3], [3, 2, 1]))
        self.assertEqual(res, 0)

    def test_get_single_point_from_which_camera(self):
        center_point_ = [0.3165423775809235, -0.10786089526290894, -0.5944049978975746]
        cur_point_ = [-4.98036103e-01, 2.93154709e-01, 2.44336074e-01]
        camera_origins = [[2.50436065, -3.75589484, 1.88800446],
                          [4.02581981, -2.56894275, -3.29281609],
                          [1.01348544, 1.88043939, -5.4273143],
                          [-2.45261002, 3.5962286, -1.87506165],
                          [-3.12155638, 2.09254542, 2.21770186],
                          [-1.07692383, -1.37631717, 4.3081322]]
        index = get_single_point_from_which_camera(center_point_, cur_point_, camera_origins)
        print("目标相机索引是：", index)

    # 测试判断点来自哪两个相机之间的函数
    def test_get_point_from_which_camera2(self):
        center_point = [-0.58192716, - 0.94316095, - 1.06762088]
        center_point_ = [0.3165423775809235, -0.10786089526290894, -0.5944049978975746]
        cur_point = [-1.261181, -2.678640, -2.523260]
        camera_origins = [[2.50436065, -3.75589484, 1.88800446],
                          [4.02581981, -2.56894275, -3.29281609],
                          [1.01348544, 1.88043939, -5.4273143],
                          [-2.45261002, 3.5962286, -1.87506165],
                          [-3.12155638, 2.09254542, 2.21770186],
                          [-1.07692383, -1.37631717, 4.3081322]]
        count = get_point_from_which_camera2(cur_point, center_point_, camera_origins)
        self.assertEqual(count, 1)

    # 计算六个相机的投影矩阵
    def test_calculator_camera_projection_mat(self):
        camera_a_projection_mat = tl.camera_a_inner_para * tl.camera_a_outer_para
        print(camera_a_projection_mat, "\n")
        camera_b_projection_mat = tl.camera_b_inner_para * tl.camera_b_outer_para
        print(camera_b_projection_mat, "\n")
        camera_c_projection_mat = tl.camera_c_inner_para * tl.camera_c_outer_para
        print(camera_c_projection_mat, "\n")
        camera_d_projection_mat = tl.camera_d_inner_para * tl.camera_d_outer_para
        print(camera_d_projection_mat, "\n")
        camera_e_projection_mat = tl.camera_e_inner_para * tl.camera_e_outer_para
        print(camera_e_projection_mat, "\n")
        camera_f_projection_mat = tl.camera_f_inner_para * tl.camera_f_outer_para
        print(camera_f_projection_mat)

    # 测试读取uv数据
    def test_read_uv_points(self):
        file_path = 'outer_files/LFMB_Visual_Hull_Meshes256/001_1_2_01'
        suffix = ".txt"
        uv_points = read_uv_points(file_path + suffix)
        tl.print_data_points(uv_points)

    # 测试用手写的函数读取位图
    def test_read_bmp(self):
        start = time.time()
        file_path = '../outer_files/images/001_1_2_01_A.bmp'
        img = rbm.read_rows(file_path)
        print(time.time() - start)
        plt.imshow(img, cmap="gray")
        plt.show()

    # 测试用cv内置库去读取位图
    def test_read_bmp2(self):
        start = time.time()
        file_path = '../outer_files/images/001_1_2_01_A.bmp'
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        print(time.time() - start)
        plt.imshow(img, cmap="gray")
        plt.show()

    def test_write_gray_to_bmp(self):
        # 此时不需要关闭文件，a+ 可读可写（末尾追加再写），文件不存在就创建，r+可读可写不存在报错
        fp = open("../outer_files/LFMB_Visual_Hull_Meshes256/001_1_2_01.txt", "a+", encoding="utf-8")
        line = fp.readline()
        fp.write("hello python1")  # \n用来换行
        fp.seek(0, 0)
        data = fp.read()
        fp.close()
        print(data)

    # 测试三维mesh与uv map（png）的纹理映射关系
    def test_uv_map_relation_mesh(self):
        uv_map_file = '../outer_files/Mesh_UVmap/saved_spot.png'
        img = cv2.imread(uv_map_file)
        pass
