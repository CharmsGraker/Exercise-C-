import tempfile

from coding.Preprocess import base_dir
from coding.utils import csv_loader, excel_loader

base_dir


# def draw_graph():
# pygraphviz
# https://blog.csdn.net/qq_43185059/article/details/111876470
import graphviz
main_branch_name = "adj_matrix_main.csv"
link_matrix_1 = csv_loader(main_branch_name)  # pd.read_excel("../table/1.xlsx", index_col=0)

branch_name = "adj_matrix_branch.csv"
link_matrix_2 = csv_loader(branch_name)  # pd.read_excel("../table/边邻接矩阵2.xlsx", index_col=0)

excel_name = 'C1附件 相关的要素名称及位置坐标数据.xls'
point_location = excel_loader(excel_name, index_col="要素编号")  # pd.read_excel("../table/相关的要素名称及位置坐标数据.xls", index_col="要素编号")

# 绘图
G = graphviz.Digraph(format='png')

# 添加节点
for point in point_location.index:
    color = "#5bc49f" if point.startswith("D") else "blue" if point.startswith("F") else "red" if point.startswith(
        "J") else "#000000"

    G.node(point,
           # shape="none",
           color=color
           # fixedsize=True,
           # width=0.3,
           # height=0.3
           # pos="{},{}!"
           # .format(0.06 * point_location.at[point, 'X坐标（单位：km）'],
           #         0.06 * (point_location.at[point, 'Y坐标（单位：km）']))
   )

# 添加边
for start in point_location.index:
    for end in point_location.index:
        # 支道
        print(start,end)
        if link_matrix_1.at[start, end] ^ link_matrix_2.at[start, end]:
            G.edge(start, end)
        # 主道
        if link_matrix_2.at[start, end]:
            G.edge(start, end, color="red", penwidth="2")

# 导出图形
src = graphviz.Source(G.source)
# G.view(tempfile.mktemp('.gv'))

# G.draw("../image/地图.png")
