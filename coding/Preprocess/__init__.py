import os

import pandas as pd
import numpy as np

from coding.Preprocess.config import InMainBranch, dataFolder_dir

from coding.Preprocess.config import base_dir

# os.path.abspath(os.getcwd())


# table_name = 'C1附件 相关的要素名称及位置坐标数据.xls'
# table_path = os.path.join('./dataFolder', table_name)
# point_name = pd.read_excel(table_path, index_col='要素编号').index


# 默认以空行分割，dropna掉
# point_name = point_name.dropna()





#
# def annotationMainOrBranch(mode=0):
#     """
#
#     :param mode: 0代表生成分支，1代表生成主干的邻接矩阵
#     :return:
#     """
#     assert mode in [0, 1]
#     # 初始化，N*N矩阵
#     net = {0: "branch", 1: "main"}
#     link_matrix = pd.DataFrame(0, index=point_name, columns=point_name)
#     for start_index, start in enumerate(point_name):
#         for end_index, end in enumerate(point_name[start_index + 1:]):
#             if input(f"L({start}-{end}) = ") == "1":
#                 link_matrix.at[start, end] = link_matrix.at[end, start] = 1
#     filename = "adj_matrix_" + net[mode] + ".xlsx"
#     save_path = os.path.join(base_dir, dataFolder_dir, "table")
#     link_matrix.to_excel(save_path)


