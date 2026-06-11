"""
项目实战：某水文站径流量预测
=========================
场景：根据气象因子预测河流径流量

数据说明（模拟5年逐月数据，共60条）：
  特征：降雨量、平均气温、蒸发量、前期径流
  目标：当月径流量 (m³/s)

用三个模型对比，看哪个最好
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# 设置随机种子，保证每次运行结果一致
np.random.seed(42)

# ============================================================
# 第一步：生成模拟水文数据（60个月 = 5年）
# ============================================================
n = 60

# 特征：降雨量 (mm/月) — 核心因子
rain = np.random.gamma(shape=8, scale=15, size=n)      # 40~200mm
# 特征：平均气温 (°C) — 影响蒸发和融雪
temp = 15 + 10 * np.sin(np.linspace(0, 10*np.pi, n)) + np.random.randn(n) * 3
# 特征：蒸发量 (mm/月)
evap = np.clip(3 + 0.3 * temp + np.random.randn(n) * 2, 0, None)
# 特征：前期径流 (m³/s) — 水文有持续性
q_last = np.zeros(n)
q_last[0] = 200
for i in range(1, n):
    q_last[i] = 150 + 0.6 * q_last[i-1] + np.random.randn() * 30

# 目标：当月径流量 (m³/s)
# 真实规律：径流 = 降雨贡献 + 前期径流贡献 - 蒸发损失 + 随机波动
q_true = (
    0.8 * rain +                    # 降雨是主要来源
    0.4 * q_last +                  # 前期径流有持续性
    -0.6 * evap +                   # 蒸发会减少径流
    0.3 * temp +                    # 气温高时融雪/冰川补给
    np.random.randn(n) * 20         # 随机误差
)
q = np.maximum(q_true, 0)           # 径流不可能为负

# 组装特征矩阵
X_raw = np.column_stack([rain, temp, evap, q_last])
feature_names = ['降雨量', '平均气温', '蒸发量', '前期径流']
y = q

print("=" * 60)
print("📊 某水文站径流预测数据集")
print("=" * 60)
print(f"样本数: {n} 个月 (5年)")
print(f"特征数: {len(feature_names)}")
for i, name in enumerate(feature_names):
    print(f"  [{i+1}] {name} 范围: {X_raw[:,i].min():.1f} ~ {X_raw[:,i].max():.1f}")
print(f"目标: 月径流量 范围: {y.min():.1f} ~ {y.max():.1f} m³/s")

# ============================================================
# 第二步：构造多项式特征 + 划分训练集/测试集
# ============================================================
# 加入特征间的交互项（比如 降雨×气温），捕捉非线性关系
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly = poly.fit_transform(X_raw)
# 列名：原始4个 + 4个平方项 + 6个交互项 = 14列
poly_names = poly.get_feature_names_out(feature_names)
n_features = X_poly.shape[1]
print(f"\n📐 构造多项式特征后: {n_features} 个特征 (含平方项和交互项)")

# 标准化（Lasso/Ridge 必须做）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# 划分训练集(80%)和测试集(20%)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)
print(f"\n📂 数据划分: 训练集 {len(y_train)} 条, 测试集 {len(y_test)} 条")

# ============================================================
# 第三步：训练三个模型
# ============================================================
print("\n" + "=" * 60)
print("🔧 模型训练")
print("=" * 60)

# --- 3a 普通线性回归 ---
lr = LinearRegression()
lr.fit(X_train, y_train)
pred_lr_train = lr.predict(X_train)
pred_lr_test = lr.predict(X_test)

# --- 3b Ridge（α=10，适中惩罚）---
ridge = Ridge(alpha=10)
ridge.fit(X_train, y_train)
pred_ridge_train = ridge.predict(X_train)
pred_ridge_test = ridge.predict(X_test)

# --- 3c Lasso（α=1，自动选特征）---
lasso = Lasso(alpha=1, max_iter=100000)
lasso.fit(X_train, y_train)
pred_lasso_train = lasso.predict(X_train)
pred_lasso_test = lasso.predict(X_test)

# 看看 Lasso 用了几个特征
n_used = np.sum(lasso.coef_ != 0)
print(f"\nLasso 从 {n_features} 个特征中选了 {n_used} 个")
for name, coef in zip(poly_names, lasso.coef_):
    if abs(coef) > 0.001:
        print(f"  ✅ {name}: {coef:.2f}")

# ============================================================
# 第四步：评估结果（重点看测试集！）
# ============================================================
print("\n" + "=" * 60)
print("📈 模型评估")
print("=" * 60)

models = [
    ("普通线性回归", lr, pred_lr_train, pred_lr_test),
    ("Ridge", ridge, pred_ridge_train, pred_ridge_test),
    ("Lasso", lasso, pred_lasso_train, pred_lasso_test),
]

results = []
for name, model, pred_tr, pred_te in models:
    rmse_tr = root_mean_squared_error(y_train, pred_tr)
    rmse_te = root_mean_squared_error(y_test, pred_te)
    r2_tr = model.score(X_train, y_train)
    r2_te = model.score(X_test, y_test)
    results.append([name, rmse_tr, rmse_te, r2_tr, r2_te])

    print(f"\n{name}:")
    print(f"  📍 训练集 → RMSE={rmse_tr:.1f}, R²={r2_tr:.3f}")
    print(f"  📍 测试集 → RMSE={rmse_te:.1f}, R²={r2_te:.3f}  ← 这个才是真本事")
    if 'coef_' in dir(model):
        max_c = max(abs(model.coef_))
        print(f"  📍 最大系数 = {max_c:.2f}")

# 找到测试集上 R² 最高的模型
best_idx = np.argmax([r[4] for r in results])
print(f"\n🏆 结论: {results[best_idx][0]} 在测试集上表现最好 (R²={results[best_idx][4]:.3f})")

# ============================================================
# 第五步：可视化
# ============================================================
plt.figure(figsize=(16, 5))

# --- 图1：测试集预测 vs 真实（最直观）---
plt.subplot(1, 3, 1)
colors = ['blue', 'red', 'green']
labels = ['线性回归', 'Ridge', 'Lasso']
preds = [pred_lr_test, pred_ridge_test, pred_lasso_test]
for i, (pred, c, label) in enumerate(zip(preds, colors, labels)):
    plt.scatter(y_test, pred, c=c, alpha=0.6, label=label, marker='o' if i==0 else 'x')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', alpha=0.5)
plt.xlabel('真实径流量 (m³/s)')
plt.ylabel('预测径流量 (m³/s)')
plt.title('测试集：预测 vs 真实')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')

# --- 图2：时间序列对比 ---
plt.subplot(1, 3, 2)
x_axis = np.arange(len(y_test))
plt.plot(x_axis, y_test, 'ko-', label='真实值', linewidth=1.5)
plt.plot(x_axis, pred_lr_test, 'bs-', label='线性回归', alpha=0.7)
plt.plot(x_axis, pred_ridge_test, 'r^--', label='Ridge', alpha=0.7)
plt.plot(x_axis, pred_lasso_test, 'gx-.', label='Lasso', alpha=0.7)
plt.xlabel('测试集样本序号')
plt.ylabel('径流量 (m³/s)')
plt.title('测试集：时间序列对比')
plt.legend()
plt.grid(True, alpha=0.3)

# --- 图3：系数对比（看谁更干净）---
plt.subplot(1, 3, 3)
x_pos = np.arange(min(14, n_features))
width = 0.25
plt.bar(x_pos - width, lr.coef_[:14], width, label='线性回归', alpha=0.6)
plt.bar(x_pos, ridge.coef_[:14], width, label='Ridge', alpha=0.6)
plt.bar(x_pos + width, lasso.coef_[:14], width, label='Lasso', alpha=0.6)
plt.axhline(y=0, color='black', linewidth=0.5)
plt.xticks(x_pos, poly_names[:14], rotation=45, fontsize=8)
plt.ylabel('系数大小')
plt.title('前14个特征的系数对比')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('水文径流预测结果.png', dpi=150, bbox_inches='tight')
print(f"\n📸 图片已保存: 水文径流预测结果.png")
print("\n✅ 项目完成！")
