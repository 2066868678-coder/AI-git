"""
8_练习题_逻辑回归
===============
真实鸢尾花数据集（二分类）

数据来源：sklearn 内置 Iris 数据集
特征：花萼长、花萼宽、花瓣长、花瓣宽（单位cm）
标签：0=不是Setosa, 1=是Setosa

要求：
  1. 创建并训练逻辑回归模型  ← 你写
  2. 输出训练集准确率         ← 你写
  3. 预测下面这朵花是不是Setosa：
     花萼长5.1cm, 花萼宽3.5cm, 花瓣长1.4cm, 花瓣宽0.2cm
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# ====== 加载真实数据（我准备）======
data = load_iris()
X = data.data                                 # 4个特征
y = (data.target == 0).astype(int)            # 1=是Setosa, 0=不是

print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
print(f"特征: {data.feature_names}")
print(f"是Setosa: {sum(y==1)}个, 不是Setosa: {sum(y==0)}个")

# ====== 你的代码从这里开始 ======
model = LogisticRegression()
model.fit(X, y)
pred = model.predict(X)
accuracy = np.mean(pred == y)
print(f"\n准确率: {accuracy * 100:.1f}%")

# 3. 预测新样本
new = np.array([[5.1, 3.5, 1.4, 0.2]])
prob = model.predict_proba(new)[0, 1]
print(f"这朵花是Setosa的概率: {prob:.1%}")

# ====== 完整评估指标 ======
tp = np.sum((pred == 1) & (y == 1))
tn = np.sum((pred == 0) & (y == 0))
fp = np.sum((pred == 1) & (y == 0))
fn = np.sum((pred == 0) & (y == 1))

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("\n" + "=" * 40)
print("分类评估指标")
print("=" * 40)
print(f"准确率 (Accuracy):  {accuracy*100:.1f}%  → 总共对了多少")
print(f"精确率 (Precision): {precision*100:.1f}%  → 我说'是Setosa'的里面，真对了几个")
print(f"召回率 (Recall):    {recall*100:.1f}%  → 真是Setosa的里面，我找出了几个")
print(f"F1分数:             {f1:.3f}         → 精确率和召回率的平衡")
print(f"\n混淆矩阵:")
print(f"{'':>12} {'预测不是':>8} {'预测是':>8}")
print(f"{'实际不是':>12} {tn:>8} {fp:>8}")
print(f"{'实际是':>12} {fn:>8} {tp:>8}")

# ====== 可视化：花瓣长 vs 花瓣宽 ======
plt.figure(figsize=(8, 6))
plt.scatter(X[y==0, 2], X[y==0, 3], color='red', s=60, label='不是Setosa', edgecolors='black')
plt.scatter(X[y==1, 2], X[y==1, 3], color='green', s=60, label='是Setosa', edgecolors='black')
plt.xlabel('花瓣长 (cm)', fontsize=12)
plt.ylabel('花瓣宽 (cm)', fontsize=12)
plt.title('鸢尾花数据 — 花瓣长 vs 花瓣宽\n两类完全可分！', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('8_练习题_可视化.png', dpi=150, bbox_inches='tight')
print("\n图片已保存: 8_练习题_可视化.png")
