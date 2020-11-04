import os
from process import process_finger_data as pfd, texture_mapping as tm

'通过面进行纹理映射'
if __name__ == '__main__':
    file_path = 'outer_files/LFMB_Visual_Hull_Meshes256/0001_1_2_01'
    obj_suffix = '.obj'
    uv_file_path = file_path + '.txt'
    # 先判断是否有生成的含像素uv的txt数据文件，如果有，直接读取txt，否则进行数据处理生成含uv的txt
    if os.path.exists(uv_file_path):
        # 读取uv文件
        uv_points = pfd.read_uv_points(uv_file_path)
        faces_texture = tm.mapping_points_gray(uv_points, file_path)  # 拿到所有面的纹理区域
        tm.write_gray_to_obj(faces_texture, file_path)
