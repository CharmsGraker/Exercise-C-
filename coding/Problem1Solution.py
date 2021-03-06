import copy
import os

import gurobipy
import pandas as pd
from matplotlib import pyplot as plt


from coding.Preprocess.costing import A_graph, B_graph

# %% 1. 数据准备
# 导入文件
from coding.draw_utils import PlotGraph, PlotPath

from utils import GraphAndCostLoader

A_graph, B_graph, A_cost_matrix, B_cost_matrix = GraphAndCostLoader()

# 初始化映射表
cost_matrix = {
    "A": A_cost_matrix,
    "B": B_cost_matrix
}

graph = {"A": A_graph, "B": B_graph}

# 车的编号
alpha = {i: "B" if i in [7, 8, 9, 10, 17, 18, 19, 20]
else "A"
         for i in range(1, 21)}

# 前10辆车在D1，后10辆在D2
beta = {i: "D1" if i in range(1, 11)
else "D2"
        for i in range(1, 21)}
# 完成洒水任务时间，换算成小时
w = {"A": 20 / 60,
     "B": 15 / 60}

# 20 辆车
I = range(1, 21)

# 喷洒作业点
J = [point for point in A_cost_matrix.index if point.startswith("F")]

# 补给点
K = [point for point in A_cost_matrix.index if point.startswith("Z")]

# %% 2. Gurobi 求解
MODEL = gurobipy.Model()

# 创建变量
x = MODEL.addVars(I, J, K, vtype=gurobipy.GRB.BINARY)
# 变量是I
t = MODEL.addVars(I)

# 目标求解变量
t_max = MODEL.addVar()

# 更新变量空间
MODEL.update()

# 创建目标函数
MODEL.setObjectiveN(t_max, priority=1, index=0)
MODEL.setObjectiveN(t.sum(), priority=0, index=1)

# 创建约束条件
# 一个洒水点只能被洒水一次
MODEL.addConstrs(sum(x[i, j, k] for j in J for k in K) == 1 for i in I)
# 一辆车只能洒水一次
MODEL.addConstrs(sum(x[i, j, k] for i in I for k in K) <= 1 for j in J)
# 一个补给点只能最多补给8次
MODEL.addConstrs(sum(x[i, j, k] for i in I for j in J) <= 8 for k in K)

# 每辆车洒水花费的时间 = 路上的时间 + 一次喷洒作业的时间
# 而路上的时间 = 前往喷洒作业点的时间 + 从喷洒点返回到补给点的时间
# alpha[i]表示车型A,B
# beta[i]表示洒水车出发的补给点
# cost_matrix[alpha[i]]得到此类车型的代价矩阵，即A_cost_matrix
# w[alpha[i]]表示喷洒作业一次花费的时间。
MODEL.addConstrs(t[i] == sum(
    (cost_matrix[alpha[i]].at[beta[i], j] + cost_matrix[alpha[i]].at[j, k]) * x[i, j, k] for j in J for k in K) + w[
                     alpha[i]] for i in I)

# 为了求解更快，这是因为肯定比单个车洒水作业时间长。
MODEL.addConstrs(t_max >= t[i] for i in I)

print('准备优化')

# 执行最优化
MODEL.optimize()

# 输出结果
print(f"任务完成用时：{round(t_max.x, 2)} h")
print(f"平均用时：{round(t.sum().getValue() / 20, 2)} h")
plt.figure()
PlotGraph()
for i in I:
    for j in J:
        for k in K:
            if x[i, j, k].x:
                path1 = graph[alpha[i]].shortest_paths_Solve(beta[i], j, show=False)[0][0][:-1]
                path2 = graph[alpha[i]].shortest_paths_Solve(j, k, show=False)[0][0]

                print_path1 = copy.deepcopy(path1)
                print_path2 = copy.deepcopy(path2)

                path2[0] = path2[0] + "(作业点)"
                path2[-1] = path2[-1] + "(补水点)"
                # print(path2)
                points = path1 + path2
                print_points = []
                # print_points.append(print_path1)
                # print_points.append(print_path2)
                # print(print_points)
                print(f"编号：{alpha[i]}-{i}用时：{round(t[i].x, 2)}ht路线：{' -> '.join(points)}")

                # PlotPath(print_points, label=f"{alpha[i]}-{[i]}")
plt.show()

