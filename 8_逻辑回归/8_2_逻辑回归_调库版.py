"""
调库版：逻辑回归（二分类）
==================
和手写版一样的数据，用 sklearn 三行搞定
"""
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# ====== 数据准备（我准备，和手写版一样）======
X = np.array([0.5, 0.8, 1.0, 1.2, 1.5, 1.8, 2.0, 2.3, 2.5, 2.8,
              3.0, 3.2, 3.5, 3.8, 4.0, 4.3, 4.5, 4.8, 5.0, 5.3])
y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
              1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

# X要reshape成二维（sklearn要求特征必须是二维数组）
X = X.reshape(-1, 1)
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X, y)
w_sk = model.coef_[0][0]
b_sk = model.intercept_[0]
print(f"sklearn 学到: w={w_sk:.4f}, b={b_sk:.4f}")
print(f"手写版学到:  w=2.2062, b=-5.0015")
# 预测
pred = model.predict(X)  # 直接出类别（0或1）
prob = model.predict_proba(X)  # 出概率（良性概率, 恶性概率）

# 看前5个样本的预测结果
print("\n前5个样本:")
print(f"  肿瘤大小: {X[:5].ravel()}")
print(f"  预测类别: {pred[:5]}")
print(f"  恶性概率: {prob[:5, 1].round(3)}")

# 准确率
accuracy = np.mean(pred == y)
print(f"\n准确率: {accuracy * 100:.1f}%")

