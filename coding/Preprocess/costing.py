import os

import pandas as pd

from coding.GraphImpl import Jgraph

# %% 1.导入数据
from coding.utils import loadingData, getDistaneMatrix
from coding.Preprocess.config import base_dir, data_save_dir

# 读取支道,主干道
link_matrix_1, link_matrix_2, point_location = loadingData()

point_name = point_location.index

# %% 2. 距离矩阵

distance = getDistaneMatrix(point_location)

# 以移动时间作为代价
A_matrix_1 = distance / 60  # A 车主干道时间
A_matrix_2 = distance / 45  # A 车支干道时间

B_matrix_1 = distance / 50  # B 车主干道时间
B_matrix_2 = distance / 30  # B 车支干道时间


# %% 3.最短路径矩阵
def shortest_matrix(matrix_1, matrix_2):
    """
    先前计算了距离矩阵，并不代表任意两点可达。而是还要参考link矩阵，link取1，距离矩阵中的value才有意义
    :param matrix_1:
    :param matrix_2:
    :return:
    """
    nodes = [{"name": i} for i in matrix_1.index]
    links = []

    for start_index, start in enumerate(point_name):
        # 遍历它之后的那些顶点，为什么?
        # 为了去重？
        for end in point_name[start_index + 1:]:
            # 加入主干道,支干道
            # 异或为1则表示， 0 1,1 0 ，主干道支干道都会被加进来
            # 这里跟原作者不太一样，因为在build的时候，已经异或过了，而不是像原作者，link1表示整张图，需要异或的到支道
            # if link_matrix_1.at[start, end] ^ link_matrix_2.at[start, end]:
            if link_matrix_1.at[start, end]:
                # print('yihuo', start, end)
                links.append({"source": start, "target": end, "value": matrix_1.at[start, end]})

            # 当前道路是主干道。
            if link_matrix_2.at[start, end]:
                # print('link_2', start, end)

                links.append({"source": start, "target": end, "value": matrix_2.at[start, end]})

    # 生成 Jgraph 图，调用最短路径算法
    graph = Jgraph(nodes, links)
    # 求出任意两点最短路径
    paths = graph.shortest_paths_Solve(point_name, point_name, 'multiple', False)

    cost_matrix = pd.DataFrame(index=point_name, columns=point_name)
    # print(cost_matrix)
    # print(paths)
    assert paths is not None
    for road in paths:
        start, end, dis = road[0][0][0], road[0][0][-1], road[0][1]
        cost_matrix.at[start, end] = dis
    return graph, cost_matrix


A_graph, A_cost_matrix = shortest_matrix(A_matrix_1, A_matrix_2)
B_graph, B_cost_matrix = shortest_matrix(B_matrix_1, B_matrix_2)
graph, cost_matrix = shortest_matrix(distance, distance)

if __name__ == '__main__':
    A_cost_matrix.to_csv(os.path.join(data_save_dir, "table/Acar_costMatrix.csv"))
    B_cost_matrix.to_csv(os.path.join(data_save_dir, "table/Bcar_costMatrix.csv"))
    # cost_matrix.to_csv()
    print('build costing done.')

