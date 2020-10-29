import unittest
from process_finger_data import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


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
        vector_product = calculate_vector_product(vector1, vector2)
        self.assertEqual(vector_product, [-4, 8, -4])

    # 测试计算向量夹角余弦值函数
    def test_calculate_cosine(self):
        vector1 = [1, 0, 0]
        vector2 = [0, 0, 1]
        res = calculate_cosine(vector1, vector2)
        print(calculate_cosine([1, 2, 3], [3, 2, 1]))
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
