import unittest
from import_finger_data import *


class Test(unittest.TestCase):

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

    # 测试判断点来自哪两个相机之间的函数
    def test_get_point_from_which_camera2(self):
        center_point = [-0.58192716, - 0.94316095, - 1.06762088]
        center_point_ = [0.18667369119257424, -0.14841140126153324, -0.5612611411749387]
        cur_point = [-0.897792 ,-0.234738, -1.299488]
        camera_origins = [[-0.59198879, 0.62321134, - 0.51103512],
                          [-0.87790013, 0.27280753, 0.39353197],
                          [-0.2893686, - 0.54638626, 0.78595666],
                          [0.42070892, - 0.8779418, 0.22852179],
                          [0.59360406, - 0.57748138, - 0.56049039],
                          [0.10318435, 0.08562591, - 0.99096982]]
        count = get_point_from_which_camera2(cur_point, center_point_, camera_origins)
        self.assertEqual(count, 1)
