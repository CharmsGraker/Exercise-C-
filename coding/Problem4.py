import gurobipy
import pandas as pd

# %% 1. 数据准备
# 导入文件
from coding.utils import getDistaneMatrix, loadingData

link_matrix_1, link_matrix_2, point_location = loadingData()
cost_matrix = getDistaneMatrix(point_location)


# 初始化映射表

Z = [point for point in cost_matrix.index if point.startswith(("J", "Z"))]
J = [point for point in cost_matrix.index if point.startswith("F")]

# %% 2. Gurobi 求解
MODEL = gurobipy.Model()

# 创建变量
y = MODEL.addVars(Z, vtype=gurobipy.GRB.BINARY)
b = MODEL.addVars(Z, J, vtype=gurobipy.GRB.BINARY)
gamma = MODEL.addVars(Z)

# 更新变量空间
MODEL.update()

# 创建目标函数
MODEL.setObjective(gamma.sum(), gurobipy.GRB.MINIMIZE)

# 创建约束条件
MODEL.addConstrs(b[z, j] <= y[z] for z in Z for j in J)
MODEL.addConstrs(sum(b[z, j] for j in J) <= 8 for z in Z)
MODEL.addConstrs(sum(b[z, j] for z in Z) == 1 for j in J)
MODEL.addConstr(sum(y[z] for z in Z) == 8)
MODEL.addConstrs(y[z] == 1 for z in [point for point in cost_matrix.index if point.startswith("Z")])
MODEL.addConstrs(gamma[z] == sum(b[z, j] * cost_matrix.at[z, j] for j in J) for z in Z)

# 执行最优化
MODEL.optimize()

# 输出结果
clusters = {z: [j for j in J if b[z, j].x] for z in Z if y[z].x}
