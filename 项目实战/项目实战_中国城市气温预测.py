"""
项目实战：哈尔滨气温预测（真实气象数据）
==========================================
数据来源：Open-Meteo 免费天气 API（无需注册）

特征：前几日气温 + 降水 + 湿度 + 蒸散 + 风速 → 预测当日最高气温

学过的知识点：
  ① 多元线性回归    ② 多项式特征（交互项）
  ③ Ridge / Lasso   ④ 训练/测试集划分（时间顺序）
  ⑤ RMSE / MAE / NSE
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import datetime
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 第一步：从 Open-Meteo 获取真实气象数据
# ============================================================
print("=" * 60)
print("🌤  中国城市气温预测（真实气象数据）")
print("=" * 60)

# 城市坐标（可换任意中国城市）
cities = {
    "北京": (39.9, 116.4),
    "上海": (31.2, 121.5),
    "广州": (23.1, 113.3),
    "武汉": (30.6, 114.3),
    "成都": (30.6, 104.1),
}

city_name = "哈尔滨"   # ← 想换城市改这里
lat, lon = cities[city_name]

print(f"\n📍 城市: {city_name} ({lat}, {lon})")

# Open-Meteo API（免费，无需key）
# 哈尔滨 45.8°N, 126.5°E | 2010-2024 共15年
url = (
    f"https://archive-api.open-meteo.com/v1/archive"
    f"?latitude={lat}&longitude={lon}"
    f"&start_date=2010-01-01&end_date=2024-12-31"
    f"&daily=temperature_2m_max,temperature_2m_min,"
    f"precipitation_sum,relative_humidity_2m_mean,"
    f"et0_fao_evapotranspiration,wind_speed_10m_max,"
    f"surface_pressure,cloud_cover_mean,dew_point_2m_mean"
    f"&timezone=Asia%2FShanghai"
)

try:
    with urllib.request.urlopen(url, timeout=15) as resp:
        raw = json.loads(resp.read().decode())

    daily = raw['daily']
    dates = np.array(daily['time'])
    tmax = np.array(daily['temperature_2m_max'], dtype=float)       # 最高气温
    tmin = np.array(daily['temperature_2m_min'], dtype=float)       # 最低气温
    precip = np.array(daily['precipitation_sum'], dtype=float)       # 降水量
    humid = np.array(daily['relative_humidity_2m_mean'], dtype=float) # 相对湿度
    et0 = np.array(daily['et0_fao_evapotranspiration'], dtype=float)  # 蒸散量
    wind = np.array(daily['wind_speed_10m_max'], dtype=float)         # 最大风速
    press = np.array(daily['surface_pressure'], dtype=float)          # 地表气压
    cloud = np.array(daily['cloud_cover_mean'], dtype=float)          # 平均云量
    dew = np.array(daily['dew_point_2m_mean'], dtype=float)           # 露点温度

    print(f"✅ 成功获取 {len(dates)} 天数据")
    print(f"   时间: {dates[0]} ~ {dates[-1]}")
    print(f"   最高气温: {tmax.min():.1f} ~ {tmax.max():.1f} °C")
    print(f"   平均降水: {precip[precip>0].mean():.1f} mm/雨天")
    print(f"   平均湿度: {humid.mean():.1f} %")
    print(f"   平均风速: {wind.mean():.1f} km/h")
    print(f"   平均蒸散: {et0.mean():.1f} mm/天")

except Exception as e:
    print(f"⚠️ 网络获取失败: {e}")
    print("📦 使用备用数据（模拟哈尔滨气象特征）...")
    np.random.seed(42)
    n = 1500
    t = np.arange(n)
    # 模拟季节性气温变化
    tmax = 10 + 16 * np.sin(2 * np.pi * t / 365 - 1.5) + np.random.randn(n) * 5
    tmin = -2 + 14 * np.sin(2 * np.pi * t / 365 - 1.5) + np.random.randn(n) * 4
    precip = np.maximum(0, np.random.exponential(3, n) * (np.random.rand(n) > 0.6))
    humid = 60 + 20 * np.random.rand(n)
    et0 = 3 + 2 * np.sin(2 * np.pi * t / 365 - 0.5) + np.random.randn(n)
    wind = 12 + 5 * np.random.rand(n)
    press = 1010 + np.random.randn(n) * 10
    cloud = 50 + 30 * np.random.rand(n)
    dew = -5 + 12 * np.sin(2 * np.pi * t / 365 - 1.5) + np.random.randn(n) * 3
    dates = np.array([str(datetime.date(2010, 1, 1) + datetime.timedelta(days=int(i)))
                      for i in range(n)])
    print(f"✅ 生成了 {n} 天的模拟气象数据")

# ============================================================
# 第二步：构造特征
# ============================================================
print("\n🔧 构造特征...")

# 目标：预测当日最高气温
y = tmax.copy()

# 滞后特征（用前几天数据预测今天）
X_list = []
feature_names = []
target = '当日最高气温'

lags_tmax = [1, 2, 3]      # 前1~3天最高温
lags_tmin = [1, 2, 3]      # 前1~3天最低温
lags_precip = [1, 2]        # 前1~2天降水
lags_humid = [1]            # 前1天湿度
lags_et0 = [1]              # 前1天蒸散
lags_wind = [1]             # 前1天风速
lags_press = [1]            # 前1天气压
lags_cloud = [1]            # 前1天云量
lags_dew = [1]              # 前1天露点

max_lag = max(max(lags_tmax), max(lags_tmin), max(lags_precip),
              max(lags_humid), max(lags_et0), max(lags_wind),
              max(lags_press), max(lags_cloud), max(lags_dew))
n = len(y)

# 逐个构造滞后特征
for lag in lags_tmax:
    X_list.append(tmax[max_lag - lag : n - lag])
    feature_names.append(f'前{lag}天最高温')
for lag in lags_tmin:
    X_list.append(tmin[max_lag - lag : n - lag])
    feature_names.append(f'前{lag}天最低温')
for lag in lags_precip:
    X_list.append(precip[max_lag - lag : n - lag])
    feature_names.append(f'前{lag}天降水')
for lag in lags_humid:
    X_list.append(humid[max_lag - lag : n - lag])
    feature_names.append(f'前{lag}天湿度')
for lag in lags_et0:
    X_list.append(et0[max_lag - lag : n - lag])
    feature_names.append(f'前{lag}天蒸散')
for lag in lags_wind:
    X_list.append(wind[max_lag - lag : n - lag])
    feature_names.append(f'前{lag}天风速')
for lag in lags_press:
    X_list.append(press[max_lag - lag : n - lag])
    feature_names.append(f'前{lag}天气压')
for lag in lags_cloud:
    X_list.append(cloud[max_lag - lag : n - lag])
    feature_names.append(f'前{lag}天云量')
for lag in lags_dew:
    X_list.append(dew[max_lag - lag : n - lag])
    feature_names.append(f'前{lag}天露点')

# 添加季节特征（月份编码）
month_of_year = np.array([int(d[5:7]) for d in dates])
month_feat = month_of_year[max_lag:]
X_list.append(np.sin(2 * np.pi * month_feat / 12))
feature_names.append('季节_正弦')
X_list.append(np.cos(2 * np.pi * month_feat / 12))
feature_names.append('季节_余弦')

# 添加温差特征（前1天最高-最低）
temp_diff = tmax - tmin
X_list.append(temp_diff[max_lag - 1 : n - 1])
feature_names.append('前1天温差')

# 目标值对齐
y = y[max_lag:]

# 合并
X_raw = np.column_stack(X_list)
n_samples = len(y)

print(f"样本数: {n_samples} 天")
print(f"特征数: {len(feature_names)}")
print(f"特征: {', '.join(feature_names[:10])}...")

# ============================================================
# 第三步：多项式 + 标准化 + 划分
# ============================================================
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_poly = poly.fit_transform(X_raw)
print(f"📐 加交互项后: {X_poly.shape[1]} 个特征")

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_poly)

# 时间顺序划分
split = int(n_samples * 0.8)
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y[:split], y[split:]
print(f"\n📂 训练集: {split} 天  | 测试集: {n_samples - split} 天")

# ============================================================
# 第四步：训练
# ============================================================
print("\n" + "=" * 60)
print("🔧 训练模型")
print("=" * 60)

lr = LinearRegression()
lr.fit(X_train, y_train)

ridge = Ridge(alpha=10)
ridge.fit(X_train, y_train)

lasso = Lasso(alpha=0.3, max_iter=100000)
lasso.fit(X_train, y_train)

n_used = np.sum(lasso.coef_ != 0)
print(f"\nLasso 从 {X_poly.shape[1]} 个特征中保留了 {n_used} 个")

# ============================================================
# 第五步：评估
# ============================================================
print("\n" + "=" * 60)
print("📈 评估结果（测试集）")
print("=" * 60)

models = [
    ("普通线性回归", lr),
    (f"Ridge (α=10)", ridge),
    (f"Lasso (α=0.3)", lasso),
]

results = []
for name, model in models:
    pred_te = model.predict(X_test)

    rmse = root_mean_squared_error(y_test, pred_te)
    mae = mean_absolute_error(y_test, pred_te)
    nse = 1 - np.sum((y_test - pred_te)**2) / np.sum((y_test - y_test.mean())**2)
    results.append([name, rmse, mae, nse])

    print(f"\n{name}:")
    print(f"  RMSE = {rmse:.1f} °C")
    print(f"  MAE  = {mae:.1f} °C")
    print(f"  NSE  = {nse:.3f}")

best_idx = np.argmax([r[3] for r in results])
best = results[best_idx]
print(f"\n🏆 {best[0]} 表现最好: MAE={best[2]:.1f}°C, NSE={best[3]:.3f}")

# ============================================================
# 第六步：可视化
# ============================================================
print("\n📸 生成可视化...")
plt.figure(figsize=(16, 10))

# --- 图1：气温时间序列 ---
plt.subplot(2, 2, 1)
plt.plot(tmax, color='orange', linewidth=0.5, label='最高温')
plt.plot(tmin, color='blue', linewidth=0.5, alpha=0.5, label='最低温')
plt.axvline(x=split + max_lag, color='red', linestyle='--', label='训练/测试分界')
plt.xlabel('天数')
plt.ylabel('气温 (°C)')
plt.title(f'{city_name} 逐日气温 (2010-2024)')
plt.legend()
plt.grid(True, alpha=0.3)

# --- 图2：预测 vs 真实散点 ---
plt.subplot(2, 2, 2)
preds = [m.predict(X_test) for _, m in models]
colors = ['blue', 'red', 'green']
labels = ['线性回归', 'Ridge', 'Lasso']
for i, (pred, c, label) in enumerate(zip(preds, colors, labels)):
    plt.scatter(y_test, pred, c=c, alpha=0.3, label=label, marker='o' if i==0 else 'x', s=15)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', alpha=0.5)
plt.xlabel('实测最高温 (°C)')
plt.ylabel('预测最高温 (°C)')
plt.title('测试集：预测 vs 实测')
plt.legend()
plt.grid(True, alpha=0.3)
# 文本框显示指标
text_str = ""
for name, rmse, mae, nse in results:
    text_str += f"{name}: RMSE={rmse:.1f}  NSE={nse:.3f}\n"
plt.text(0.05, 0.95, text_str, transform=plt.gca().transAxes,
         fontsize=8, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# --- 图3：测试集最后90天 ---
plt.subplot(2, 2, 3)
last_n = 90
x_axis = np.arange(last_n)
plt.plot(x_axis, y_test[-last_n:], 'k-', label='实测', linewidth=2)
plt.plot(x_axis, lr.predict(X_test)[-last_n:], 'b-', label='线性回归', alpha=0.7)
plt.plot(x_axis, ridge.predict(X_test)[-last_n:], 'r--', label='Ridge', alpha=0.7)
plt.plot(x_axis, lasso.predict(X_test)[-last_n:], 'g-.', label='Lasso', alpha=0.7)
plt.xlabel('天数')
plt.ylabel('最高气温 (°C)')
plt.title(f'测试集最后{last_n}天：预测曲线对比')
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
plt.xticks(x_pos, poly.get_feature_names_out(feature_names)[:n_show],
           rotation=45, fontsize=7)
plt.ylabel('系数大小')
plt.title('系数对比')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('项目实战_中国城市气温预测.png', dpi=150, bbox_inches='tight')
print("✅ 图片已保存: 项目实战_中国城市气温预测.png")
print(f"\n🎉 项目完成！当前城市: {city_name}，想换城市改第28行的 city_name 即可")
