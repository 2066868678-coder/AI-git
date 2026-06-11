"""
逻辑回归 — 真实糖尿病预测
====================
Pima Indians 数据集，根据体检指标判断是否患糖尿病
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# ====== 加载真实数据 ======
print("正在下载数据...")
data = fetch_openml('diabetes', version=1, as_frame=False, parser='auto')
X = data.data
y = (data.target == 'tested_positive').astype(int)

feature_names = ['怀孕次数', '血糖', '血压', '皮褶厚', '胰岛素', 'BMI', '遗传函数', '年龄']

print(f"样本数: {X.shape[0]}")
print(f"特征数: {X.shape[1]}")
print(f"标签:   无糖尿病 {sum(y==0)}个, 有糖尿病 {sum(y==1)}个")

print("\n各特征统计：")
print(f"{'特征':<10} {'平均':>8} {'标准差':>8} {'最小':>8} {'最大':>8}")
print("-" * 42)
for i, name in enumerate(feature_names):
    print(f"{name:<10} {X[:,i].mean():>8.1f} {X[:,i].std():>8.1f} {X[:,i].min():>8.0f} {X[:,i].max():>8.0f}")
