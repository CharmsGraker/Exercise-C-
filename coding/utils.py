import os

import pandas as pd

from coding.Preprocess.config import data_save_dir, base_dir, InMainBranch, dataFolder_dir


def excel_loader(filename, index_col):
    table = pd.read_excel(os.path.join(data_save_dir, filename), engine='xlrd', index_col=index_col)
    table = table.dropna()
    return table


def csv_loader(filename, index_col=0):
    table = pd.read_csv(os.path.join(data_save_dir, 'table', filename), index_col=index_col)
    return table


def loadingData():
    """

    :return: 返回支道,主干道,顶点
    """

    branch_name = "adj_matrix_branch.csv"
    link_matrix_1 = csv_loader(branch_name)  # pd.read_excel("../table/边邻接矩阵2.xlsx", index_col=0)

    main_branch_name = "adj_matrix_main.csv"
    link_matrix_2 = csv_loader(main_branch_name)  # pd.read_excel("../table/1.xlsx", index_col=0)

    excel_name = 'C1附件 相关的要素名称及位置坐标数据.xls'
    point_location = excel_loader(excel_name,
                                  index_col="要素编号")  # pd.read_excel("../table/相关的要素名称及位置坐标数据.xls", index_col="要素编号")
    return link_matrix_1, link_matrix_2, point_location


def getDistaneMatrix(point_location):
    """
    计算任意两点的距离，得到距离矩阵
    :param point_location:
    :return:
    """
    point_name = point_location.index
    distance = pd.DataFrame(0.0, index=point_name, columns=point_name)

    for start in point_name:
        for end in point_name:
            x1, y1 = point_location.at[start, 'X坐标（单位：km）'], point_location.at[start, 'Y坐标（单位：km）']
            x2, y2 = point_location.at[end, 'X坐标（单位：km）'], point_location.at[end, 'Y坐标（单位：km）']
            distance.at[start, end] = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1 / 2)
    return distance



def getLocation():
    excel_name = 'C1附件 相关的要素名称及位置坐标数据.xls'
    point_location = excel_loader(excel_name, index_col="要素编号")
    return point_location


def annotationMain():
    """

    :param mode: 0代表生成分支，1代表生成主干的邻接矩阵
    :return:
    """
    point_location = getLocation()
    point_name = point_location.index

    # 初始化，N*N矩阵
    net = {0: "branch", 1: "main"}
    link_matrix_main = pd.DataFrame(0, index=point_name, columns=point_name)
    for start_index, start in enumerate(point_name):
        for end_index, end in enumerate(point_name[start_index + 1:]):
            if [start, end] in InMainBranch or [end, start] in InMainBranch:
                link_matrix_main.at[start, end] = link_matrix_main.at[end, start] = 1
    filename = "adj_matrix_" + net[1] + ".csv"

    save_dir = os.path.join(base_dir, dataFolder_dir, "table")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path_main = os.path.join(save_dir, filename)

    link_matrix_main.to_csv(save_path_main)

    link_matrix_branch = 1 ^ link_matrix_main

    filename = "adj_matrix_" + net[0] + ".csv"
    save_path_branch = os.path.join(save_dir, filename)

    link_matrix_branch.to_csv(save_path_branch)




