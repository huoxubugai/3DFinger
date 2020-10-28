import import_finger_data as ifd
import numpy as np

if __name__ == '__main__':
    objFilePath = 'D:/PythonCode/3DFingers/LFMB_Visual_Hull_Meshes256/001_1_2_01.obj'
    # 拿到mesh所有顶点数据
    data_points = ifd.get_mesh_point(objFilePath)  # 数据点的数据结构选择list而不是数组,方便后续改动
    print("原始数据点为：", data_points)
    # 求出所有顶点对应的中心点O
    center_point = ifd.get_center_point(data_points)
    print("中心点是：\n", center_point)
    # 求出六个相机在世界坐标系下的原点
    camera_points = ifd.get_all_camera_origin()
    print("六个相机的原点为：\n", camera_points)
    # 获取相机平面的参数ax+by+cz+d=0
    camera_plane_para = ifd.get_camera_plane(camera_points)
    print("六个相机的平面参数为：", camera_plane_para)
    # 获取O，A，E，F的映射点
    center_point_mapping = ifd.get_mapping_point_in_camera_plane(center_point, camera_plane_para)
    print("映射后的中心点O坐标：\n", center_point_mapping)
    camera_a_point = ifd.get_mapping_point_in_camera_plane(camera_points[0], camera_plane_para)
    camera_e_point = ifd.get_mapping_point_in_camera_plane(camera_points[4], camera_plane_para)
    camera_f_point = ifd.get_mapping_point_in_camera_plane(camera_points[5], camera_plane_para)
    # 六个相机归到一个平面之后的坐标：BCD不变，AEF映射到BCD平面
    camera_point_mapping = [camera_a_point, camera_points[1], camera_points[2],
                            camera_points[3], camera_e_point, camera_f_point]
    camera_point_mapping = np.array(camera_point_mapping)
    print("映射后的相机坐标：\n", camera_point_mapping)
    # 将mesh顶点数据中的所有顶点映射到相机平面
    data_points_mapping = ifd.get_data_points_mapping(data_points, camera_plane_para)
    print("映射后的所有数据点：\n",data_points_mapping)
    # ifd.print_data_points(data_points_mapping)

    # 数据预处理完毕，寻找每个点对应的相机
    data_points_contain_camera = ifd.get_data_points_from_which_camera(center_point_mapping, data_points_mapping,
                                                                       camera_point_mapping)
    print("获取所有数据点以及其来源的相机索引\n")
    ifd.print_data_points(data_points_contain_camera)
