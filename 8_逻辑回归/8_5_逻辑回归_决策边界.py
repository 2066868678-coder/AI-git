"""
逻辑回归 — 决策边界可视化
====================
只选2个特征，画出分类线
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# ====== 只取2个特征：半径 和 纹理 ======
data = load_breast_cancer()
X = data.data[:, [0, 1]]   # 只取前2列
y = data.target

# 训练逻辑回归
model = LogisticRegression()
model.fit(X, y)

# 用训练好的模型预测
pred = model.predict(X)
accuracy = np.mean(pred == y)

print(f"只用2个特征（半径+纹理），准确率: {accuracy*100:.1f}%")
print(f"模型参数: w1={model.coef_[0,0]:.3f}, w2={model.coef_[0,1]:.3f}, b={model.intercept_[0]:.3f}")

# ====== 画决策边界 ======
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 200),
                       np.linspace(x2_min, x2_max, 200))
Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
Z = Z.reshape(xx1.shape)

plt.figure(figsize=(10, 8))
plt.contourf(xx1, xx2, Z, alpha=0.3, cmap='RdYlGn')
plt.scatter(X[y==0, 0], X[y==0, 1], color='red', s=50, label='恶性', edgecolors='black')
plt.scatter(X[y==1, 0], X[y==1, 1], color='green', s=50, label='良性', edgecolors='black')

plt.xlabel('肿瘤半径', fontsize=12)
plt.ylabel('肿瘤纹理', fontsize=12)
plt.title('逻辑回归决策边界 — 半径 vs 纹理', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('8_5_决策边界.png', dpi=150, bbox_inches='tight')
print("\n图片已保存: 8_5_决策边界.png")
