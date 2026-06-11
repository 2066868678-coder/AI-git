"""
项目实战：基于USGS真实水文数据的径流预测
=====================================
数据来源：美国地质调查局(USGS)公开API
站点：波托马克河 Potomac River near Washington, DC (站点号: 01646500)
特征：历史径流 + 季节因素 → 预测当日径流

学过的知识点全覆盖：
  ① 多元线性回归
  ② 多项式特征（捕捉非线性）
  ③ Ridge / Lasso 正则化（防过拟合）
  ④ 训练集/测试集划分
  ⑤ RMSE / R² 评估
"""
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
import urllib.request
import datetime
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

print("=" * 60)
print("🌊 波托马克河径流预测（真实数据）")
print("=" * 60)

# ============================================================
# 第一步：从 USGS 获取真实水文数据
# ============================================================
print("\n📡 正在从USGS获取真实水文数据...")

# USGS API 参数
site_no = "01646500"           # 波托马克河近DC段
param = "00060"                # 参数代码：00060 = 径流(cubic feet per second)
start = "2018-01-01"           # 起始日期
end = "2023-12-31"             # 结束日期（6年数据）

url = (
    f"https://waterservices.usgs.gov/nwis/dv/"
    f"?format=json&sites={site_no}"
    f"&parameterCd={param}"
    f"&startDT={start}&endDT={end}"
    f"&siteStatus=all"
)

try:
    with urllib.request.urlopen(url, timeout=15) as resp:
        raw = json.loads(resp.read().decode())

    # 解析USGS返回的JSON数据
    ts = raw['value']['timeSeries'][0]
    values = ts['values'][0]['value']

    # 提取日期和径流值，去除缺失数据
    dates_raw = []
    q_raw = []
    for v in values:
        if v.get('qualifiers', [''])[0] != 'Ice':  # 去掉冰期异常数据
            dates_raw.append(v['dateTime'])
            q_raw.append(float(v['value']))

    # 转成numpy数组
    discharge = np.array(q_raw)
    dates = np.array(dates_raw)

    print(f"✅ 成功获取 {len(discharge)} 天数据")
    print(f"   时间范围: {dates[0]} ~ {dates[-1]}")
    discharge = discharge * 0.0283168  # cfs → m^3/s
    print(f"   径流范围: {discharge.min():.0f} ~ {discharge.max():.0f} m^3/s")
    print(f"   平均径流: {discharge.mean():.0f} m^3/s")

    USE_REAL_DATA = True

except Exception as e:
    print(f"⚠️ 网络获取失败 ({e})")
    print("📦 使用备用数据...")
    # 备用：模拟类似波托马克河的水文数据
    np.random.seed(42)
    n = 1500
    t = np.arange(n)
    # 模拟季节性 + 随机波动 + 自相关
    seasonal = 50000 + 30000 * np.sin(2 * np.pi * t / 365 + 1)
    noise = np.random.randn(n) * 5000
    ar = np.zeros(n)
    ar[0] = 50000
    for i in range(1, n):
        ar[i] = 0.7 * ar[i-1] + np.random.randn() * 3000
    discharge = seasonal + ar * 0.3 + noise
    discharge = np.maximum(discharge, 5000)
    dates = np.array([str(datetime.date(2019, 1, 1) +
                         datetime.timedelta(days=int(i))) for i in range(n)])
    print(f"✅ 生成了 {len(discharge)} 天的模拟径流数据")

    USE_REAL_DATA = False

# ============================================================
# 第二步：特征工程
# ============================================================
print("\n🔧 构造特征...")

# 提取日期信息
days_from_start = np.arange(len(discharge))
# 月份特征（用于捕捉季节规律）
month_of_year = np.array([int(d[5:7]) for d in dates])

# 构造滞后特征（用前几天的径流预测今天的）
def make_lag_features(data, lags):
    """构造滞后特征矩阵"""
    n = len(data)
    max_lag = max(lags)
    X = np.zeros((n - max_lag, len(lags)))
    y = data[max_lag:]
    for i, lag in enumerate(lags):
        X[:, i] = data[max_lag - lag : n - lag]
    return X, y

# 滞后：前1天、前2天、前3天、前7天、前14天
lags = [1, 2, 3, 7, 14]
X_lag, y = make_lag_features(discharge, lags)

# 添加月份特征（季节因素）
month_feat = month_of_year[max(lags):]  # 对齐
# 添加前7天滚动平均作为特征
roll_mean_7 = np.convolve(discharge, np.ones(7)/7, mode='valid')
# roll_mean_7[k] = 第k到k+6天的平均 → 第k+3天为中心
# y[t] = discharge[t+14]，其前7天平均 = roll_mean_7[t+7]
n_samples = len(y)
roll_mean_7 = roll_mean_7[max(lags)-7 : max(lags)-7 + n_samples]  # 对齐到y的长度

X_extra = np.column_stack([
    np.sin(2 * np.pi * month_feat / 12),   # 月份的正弦编码（季节循环）
    np.cos(2 * np.pi * month_feat / 12),    # 月份的余弦编码
    roll_mean_7,                             # 7天滚动平均
])

# 合并所有特征
X_full = np.column_stack([X_lag, X_extra])

feature_names = [
    '前1天径流', '前2天径流', '前3天径流',
    '前7天径流', '前14天径流',
    '季节_正弦', '季节_余弦', '7天滚动平均'
]

n_samples = len(y)
print(f"样本数: {n_samples}")
print(f"特征数: {len(feature_names)}")
for i, name in enumerate(feature_names):
    print(f"  [{i+1}] {name}")

# ============================================================
# 第三步：多项式特征 + 标准化 + 划分数据集
# ============================================================
# 加入交互项（例如：前1天×前7天，前1天×季节等）
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly = poly.fit_transform(X_full)
poly_names = poly.get_feature_names_out(feature_names)
print(f"\n📐 加交互项后: {X_poly.shape[1]} 个特征")

# 标准化（Ridge/Lasso必需）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# 时间序列划分：前80%训练，后20%测试
# 注意：时间序列不能用随机划分，必须按时间顺序！
split = int(n_samples * 0.8)
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y[:split], y[split:]

print(f"\n📂 数据划分（按时间顺序）:")
print(f"   训练集: {len(y_train)} 天 ({dates[max(lags)]} ~ {dates[max(lags)+split-1]})")
print(f"   测试集: {len(y_test)} 天 ({dates[max(lags)+split]} ~ {dates[-1]})")

# ============================================================
# 第四步：训练三个模型
# ============================================================
print("\n" + "=" * 60)
print("🔧 训练模型")
print("=" * 60)

# 4a: 普通线性回归
lr = LinearRegression()
lr.fit(X_train, y_train)

# 4b: Ridge
ridge = Ridge(alpha=10)
ridge.fit(X_train, y_train)

# 4c: Lasso（自动选特征）
lasso = Lasso(alpha=0.5, max_iter=100000)
lasso.fit(X_train, y_train)

# Lasso用了几个特征
n_used = np.sum(lasso.coef_ != 0)
print(f"\nLasso 从 {X_poly.shape[1]} 个特征中保留了 {n_used} 个")
# 打印被Lasso选中的重要特征
important_idx = np.argsort(np.abs(lasso.coef_))[::-1][:8]
print("Lasso认为最重要的特征:")
for idx in important_idx:
    if abs(lasso.coef_[idx]) > 0.001:
        print(f"  ✅ {poly_names[idx]}: {lasso.coef_[idx]:.2f}")

# ============================================================
# 第五步：评估（重点看测试集！）
# ============================================================
print("\n" + "=" * 60)
print("📈 评估结果")
print("=" * 60)

models = [
    ("普通线性回归", lr),
    ("Ridge (α=10)", ridge),
    ("Lasso (α=0.5)", lasso),
]

results = []
for name, model in models:
    pred_tr = model.predict(X_train)
    pred_te = model.predict(X_test)

    rmse_te = root_mean_squared_error(y_test, pred_te)
    mae_te = mean_absolute_error(y_test, pred_te)
    # NSE = Nash-Sutcliffe 效率系数（水文最核心指标，≈1最好，<0不如直接用均值）
    nse = 1 - np.sum((y_test - pred_te)**2) / np.sum((y_test - y_test.mean())**2)
    r2_te = model.score(X_test, y_test)
    results.append([name, rmse_te, mae_te, nse])

    print(f"\n{name}:")
    print(f"  RMSE = {rmse_te:.0f} m³/s  （均方根误差）")
    print(f"  MAE  = {mae_te:.0f} m³/s  （平均绝对误差）")
    print(f"  NSE  = {nse:.3f}  （Nash系数，越接近1越好）")

best_idx = np.argmax([r[3] for r in results])
best = results[best_idx]
print(f"\n🏆 结论: {best[0]} 测试集表现最好")
print(f"   RMSE = {best[1]:.0f} m³/s | MAE = {best[2]:.0f} m³/s | NSE = {best[3]:.3f}")

if USE_REAL_DATA:
    print("\n📌 数据来源: USGS National Water Information System")
else:
    print("\n📌 注: 当前使用模拟数据，网络正常后可自动获取真实数据")

# ============================================================
# 第六步：可视化
# ============================================================
print("\n📸 生成可视化...")

plt.figure(figsize=(16, 10))

# --- 图1：整段时间序列 ---
plt.subplot(2, 2, 1)
plt.plot(discharge, color='steelblue', linewidth=0.5)
plt.axvline(x=split + max(lags), color='red', linestyle='--', label='训练/测试分界')
plt.xlabel('天数')
plt.ylabel('径流 (m^3/s)')
plt.title('波托马克河日径流时间序列 (2018-2023)')
plt.legend()
plt.grid(True, alpha=0.3)

# --- 图2：测试集预测 vs 真实 ---
plt.subplot(2, 2, 2)
preds = [lr.predict(X_test), ridge.predict(X_test), lasso.predict(X_test)]
colors = ['blue', 'red', 'green']
labels = ['线性回归', 'Ridge', 'Lasso']
markers = ['o', 'x', '^']
for i, (pred, c, label, m) in enumerate(zip(preds, colors, labels, markers)):
    plt.scatter(y_test, pred, c=c, alpha=0.4, label=label, marker=m, s=20)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', alpha=0.5)
plt.xlabel('真实径流 (m^3/s)')
plt.ylabel('预测径流 (m^3/s)')
plt.title('测试集：预测 vs 真实')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
# 在左上角贴评估结果
text_str = ""
for name, rmse, mae, nse in results:
    text_str += f"{name}: RMSE={rmse:.0f}  NSE={nse:.3f}\n"
plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes,
         fontsize=8, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# --- 图3：测试集最后90天时间序列对比 ---
plt.subplot(2, 2, 3)
last_n = 90
x_axis = np.arange(last_n)
plt.plot(x_axis, y_test[-last_n:], 'k-', label='真实值', linewidth=2)
plt.plot(x_axis, lr.predict(X_test)[-last_n:], 'b-', label='线性回归', alpha=0.7)
plt.plot(x_axis, ridge.predict(X_test)[-last_n:], 'r--', label='Ridge', alpha=0.7)
plt.plot(x_axis, lasso.predict(X_test)[-last_n:], 'g-.', label='Lasso', alpha=0.7)
plt.xlabel('天数')
plt.ylabel('径流 (m^3/s)')
plt.title('测试集最后90天：预测曲线对比')
plt.legend()
plt.grid(True, alpha=0.3)

# --- 图4：系数对比 ---
plt.subplot(2, 2, 4)
n_show = min(15, X_poly.shape[1])
x_pos = np.arange(n_show)
width = 0.25
plt.bar(x_pos - width, lr.coef_[:n_show], width, label='线性回归', alpha=0.6)
plt.bar(x_pos, ridge.coef_[:n_show], width, label='Ridge', alpha=0.6)
plt.bar(x_pos + width, lasso.coef_[:n_show], width, label='Lasso', alpha=0.6)
plt.axhline(y=0, color='black', linewidth=0.5)
plt.xticks(x_pos, poly_names[:n_show], rotation=45, fontsize=7)
plt.ylabel('系数大小')
plt.title(f'前{n_show}个特征的系数对比')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('项目实战_真实水文预测.png', dpi=150, bbox_inches='tight')
print("✅ 图片已保存: 项目实战_真实水文预测.png")
print("\n🎉 项目完成！")
