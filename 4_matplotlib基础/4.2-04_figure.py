import matplotlib.pyplot as plt
import numpy as np

# # 创建画布
# fig = plt.figure(figsize=(8, 6))
#
# # 左图：合并第1格和第3格（第一列）
# ax_left = fig.add_subplot(2, 2, (1, 3))
#
# # 右上图：第2格
# ax_top_right = fig.add_subplot(2, 2, 2)
#
# # 右下图：第4格
# ax_bottom_right = fig.add_subplot(2, 2, 4)
#
# # 在左图画正弦曲线
# x = np.linspace(0, 10, 100)
# ax_left.plot(x, np.sin(x), color='blue')
# ax_left.set_title('Sine Wave (Left)')
#
# # 在右上图画散点图
# x_scatter = np.random.rand(30)
# y_scatter = np.random.rand(30)
# ax_top_right.scatter(x_scatter, y_scatter, color='red')
# ax_top_right.set_title('Scatter (Top Right)')
#
# # 在右下图画柱状图
# categories = ['A', 'B', 'C']
# values = [3, 7, 5]
# ax_bottom_right.bar(categories, values, color='green')
# ax_bottom_right.set_title('Bar (Bottom Right)')
#
# plt.tight_layout()
# plt.show()
# =========================================================================
# =========================================================================
fig =plt.figure(figsize=(8, 6))
ax_left = fig.add_subplot(2,2(1, 3))
x_line = np.array([-3, -2, -1, 0, 1, 2, 3])
y_line = np.array([9, 4, 1, 0, 1, 4, 9])
ax_left.plot(x_line,y_line)