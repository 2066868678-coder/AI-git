"""
逻辑回归 — 真实乳腺癌数据集
=====================
sklearn 内置数据，30个特征判断良性/恶性
你用模型代码，数据我准备
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# ====== 加载真实数据 ======
data = load_breast_cancer()
X = data.data        # 30个特征
y = data.target      # 0=恶性, 1=良性

# 特征中文名对照
feature_names_cn = [
    '半径', '纹理', '周长', '面积', '光滑度',
    '紧密度', '凹陷度', '凹陷点', '对称性', '分形维数',
    '半径_标准差', '纹理_标准差', '周长_标准差', '面积_标准差', '光滑度_标准差',
    '紧密度_标准差', '凹陷度_标准差', '凹陷点_标准差', '对称性_标准差', '分形维数_标准差',
    '半径_最差', '纹理_最差', '周长_最差', '面积_最差', '光滑度_最差',
    '紧密度_最差', '凹陷度_最差', '凹陷点_最差', '对称性_最差', '分形维数_最差'
]

print("=" * 50)
print("乳腺癌数据集（Wisconsin）")
print("=" * 50)
print(f"样本数: {X.shape[0]}")
print(f"特征数: {X.shape[1]}")
print(f"标签:   0=恶性 {sum(y==0)}个, 1=良性 {sum(y==1)}个")
print(f"\n30个特征（前10个特征的均值对比）：")
print(f"{'特征':<12} {'良性平均':>8} {'恶性平均':>8}")
print("-" * 30)
for i in range(10):
    benign_mean = X[y==1, i].mean()
    malignant_mean = X[y==0, i].mean()
    print(f"{feature_names_cn[i]:<12} {benign_mean:>8.2f} {malignant_mean:>8.2f}")
model = LogisticRegression(max_iter= 5000)
model.fit(X, y)
pred  = model.predict(X)
accuracy = np.mean(pred == y)
print(f"\n训练集准确率: {accuracy * 100:.2f}%")
print(f"    (569个样本，判断对了 {int(accuracy * 569)} 个)")

# ====== 可视化：混淆矩阵 ======
# 算清楚：哪些分对了，哪些分错了
tp = np.sum((pred == 1) & (y == 1))   # 良性→判良性 ✓
tn = np.sum((pred == 0) & (y == 0))   # 恶性→判恶性 ✓
fp = np.sum((pred == 1) & (y == 0))   # 恶性→判良性 ✗（误诊）
fn = np.sum((pred == 0) & (y == 1))   # 良性→判恶性 ✗（漏诊）

print("\n" + "=" * 40)
print("混淆矩阵")
print("=" * 40)
print(f"{'':>15} {'预测恶性':>8} {'预测良性':>8}")
print(f"{'实际恶性':>15} {tn:>8} {fp:>8}")
print(f"{'实际良性':>15} {fn:>8} {tp:>8}")
print("-" * 40)
print(f"准确率: {accuracy*100:.1f}%  ({tn+tp}/{tn+fp+fn+tp})")

# 画成图
plt.figure(figsize=(6, 5))
matrix = np.array([[tn, fp], [fn, tp]])
plt.imshow(matrix, cmap='Blues', alpha=0.7)
for i in range(2):
    for j in range(2):
        plt.text(j, i, str(matrix[i, j]),
                 ha='center', va='center', fontsize=20, fontweight='bold')
        plt.text(j, i+0.25, f'{matrix[i, j]/569*100:.1f}%',
                 ha='center', va='center', fontsize=11, color='gray')

plt.xticks([0, 1], ['预测恶性', '预测良性'], fontsize=12)
plt.yticks([0, 1], ['实际恶性', '实际良性'], fontsize=12)
plt.xlabel('预测结果', fontsize=13)
plt.ylabel('真实标签', fontsize=13)
plt.title('混淆矩阵 — 乳腺癌分类结果', fontsize=14, fontweight='bold')

# 加文字说明
plt.figtext(0.5, -0.05,
            f'正确: {tn+tp}个  错误: {fp+fn}个  准确率: {accuracy*100:.1f}%',
            ha='center', fontsize=12)
plt.tight_layout()
plt.savefig('8_3_混淆矩阵.png', dpi=150, bbox_inches='tight')
print("\n图片已保存: 8_3_混淆矩阵.png")

