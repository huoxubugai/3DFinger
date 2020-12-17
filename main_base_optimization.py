# -*- coding: utf-8 -*-
"""
@Time ： 2020/12/17 10:33
@Auth ： 零分
@File ：main_base_optimization.py
@IDE ：PyCharm
@github:https://github.com/huoxubugai/3DFinger
"""
import main_base_face
import os

if __name__ == '__main__':
    dir_path = 'outer_files/LFMB_Visual_Hull_Meshes256'
    path_list = os.listdir(dir_path)
    total_loss = 0
    for path in path_list:
        path_str = path.split(".")
        cur_file_path = dir_path + '/' + path_str[0]
        cur_loss = main_base_face.main(cur_file_path)
        total_loss += cur_loss
    print(total_loss)