import matplotlib.pyplot as plt
import numpy as np

# 模拟数据：10天的降雨量 vs 土壤湿度
x = np.array([60, 75, 80, 90, 55])
y = np.array([65, 70, 85, 95, 50])
plt.scatter(x, y, color = 'blue', s= 80)
plt.show()
