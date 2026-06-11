import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
plt.rcParams['font.sans-serif'] = ['SimHei']     # 使用黑体显示中文
plt.rcParams['axes.unicode_minus'] = False       # 正常显示负号
X, y = load_diabetes(return_X_y= True)
X =  X[:,2].reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2 ,random_state=42)
w = 0
b = 0
lr = 0.1
epochs = 100
loss_history = []
for i in range(epochs):
    y_pred = w * X_train + b
    loss = np.mean((y_pred - y_train) ** 2)
    loss_history.append(loss)
    dw = np.mean((y_pred-y_train)*X_train.ravel())
    db = np.mean(y_pred-y_train)
    w-= lr*dw
    b-= lr*db
print(f"最终: w={w:.4f}, b={b:.4f}")
y_pred_test = w * X_test + b
# 损失函数可视化
plt.plot(loss_history)
min_loss = min(loss_history)
min_index = loss_history.index(min_loss)
plt.text(min_index,min_loss, f"min = {min_loss:.2f}")
plt.xlabel(f'迭代次数:{epochs}学习率:{lr}')
plt.ylabel(f'损失值')
plt.title('训练过程中的损失下降曲线')
plt.show()
# 回归模型可视化
plt.scatter(X_test,y_test)
plt.plot(X_test, y_pred_test)
plt.show()
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

# 计算测试集上的预测值（你已经有了 y_pred_test）
# y_pred_test = w * X_test + b   # 这行你已经有了

# 皮尔逊相关系数（使用 numpy 计算，不依赖 scipy）
corr = np.corrcoef(X_test.ravel(), y_test)[0, 1]

# 决定系数 R²
r2 = r2_score(y_test, y_pred_test)

# 均方误差 MSE
mse = mean_squared_error(y_test, y_pred_test)

# 平均绝对误差 MAE
mae = mean_absolute_error(y_test, y_pred_test)

print(f"皮尔逊相关系数 r: {corr:.4f}")
print(f"决定系数 R²: {r2:.4f}")
print(f"均方误差 MSE: {mse:.2f}")
print(f"平均绝对误差 MAE: {mae:.2f}")
# 123