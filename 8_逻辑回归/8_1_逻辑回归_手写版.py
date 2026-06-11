"""
手写逻辑回归（二分类）
=================
用肿瘤大小判断良性/恶性
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# ====== 数据准备（我准备）======
# 肿瘤大小（cm）
X = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.3, 2.5, 2.8,
              3.0, 3.2, 3.5, 3.8, 4.0, 4.3, 4.5, 4.8, 5.0, 5.3])
# 标签：0=良性，1=恶性
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

print("样本数:", len(X))
print("肿瘤大小:", X)
print("标签(0良性/1恶性):", y)

# ====== Sigmoid 函数 ======
# 作用：把任意数值压到0~1之间，输出的是"是恶性"的概率
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
w = 0
b = 0
z = w*X + b
p= sigmoid(z)
print("初始预测概率（前10个）:", p[:10].round(3))
# ====== 训练循环 ======
learning_rate = 0.1
loss_history = []                                   # 记录每一步的损失

for i in range(1000):
    z = w * X + b
    p = sigmoid(z)
    loss = -np.mean(y * np.log(p + 1e-10) + (1 - y) * np.log(1 - p + 1e-10))
    loss_history.append(loss)                       # 存下来

    dw = np.mean((p - y) * X)
    db = np.mean(p - y)

    w = w - learning_rate * dw
    b = b - learning_rate * db

    if i % 200 == 0:
        print(f"第{i:4d}步 → loss={loss:.4f}, w={w:.4f}, b={b:.4f}")

print(f"\n训练完成！最终 w={w:.4f}, b={b:.4f}")

# ====== 可视化：损失下降曲线 ======
plt.figure(figsize=(8, 5))
plt.plot(loss_history, 'b-', linewidth=1.5)
plt.xlabel('训练步数')
plt.ylabel('损失 (Log Loss)')
plt.title('损失下降曲线 — 逻辑回归训练过程')
plt.grid(True, alpha=0.3)
plt.show()


