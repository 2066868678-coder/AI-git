"""
阈值调优 — 在漏报和空报之间找平衡
============================
逻辑回归默认用0.5做阈值，但可以调高调低
调高 → 暴雨预测更谨慎（空报↓，但可能漏报↑）
调低 → 暴雨预测更敏感（漏报↓，但空报↑）

水利上：调高还是调低，看你是防汛还是水库调度
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request, json
from sklearn.linear_model import LogisticRegression

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# ====== 加载数据 ======
url = ('https://archive-api.open-meteo.com/v1/archive?'
       'latitude=39.9&longitude=116.4'
       '&start_date=2015-01-01&end_date=2024-12-31'
       '&daily=precipitation_sum,temperature_2m_max,temperature_2m_min,'
       'relative_humidity_2m_mean,pressure_msl_mean,cloud_cover_mean,'
       'wind_speed_10m_max&timezone=Asia/Shanghai')
resp = urllib.request.urlopen(url, timeout=15)
daily = json.loads(resp.read())['daily']

df = pd.DataFrame({
    '降雨量': daily['precipitation_sum'], '最高温': daily['temperature_2m_max'],
    '最低温': daily['temperature_2m_min'], '湿度': daily['relative_humidity_2m_mean'],
    '气压': daily['pressure_msl_mean'], '云量': daily['cloud_cover_mean'],
    '风速': daily['wind_speed_10m_max'],
}).dropna()
df['暴雨'] = (df['降雨量'] >= 25).astype(int)

X = df[['湿度', '最低温', '风速', '气压']].values
y = df['暴雨'].values
split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ====== 训练模型 ======
model = LogisticRegression(max_iter=2000, class_weight='balanced')
model.fit(X_train, y_train)
prob = model.predict_proba(X_test)[:, 1]

# ====== 试不同阈值 ======
thresholds = [0.3, 0.5, 0.7, 0.9]
print(f"{'阈值':>6} {'准确率':>8} {'精确率':>8} {'召回率':>8} {'空报次数':>8} {'漏报次数':>8}")
print("-" * 55)

results = []
for t in thresholds:
    pred = (prob >= t).astype(int)
    tp = np.sum((pred == 1) & (y_test == 1))
    tn = np.sum((pred == 0) & (y_test == 0))
    fp = np.sum((pred == 1) & (y_test == 0))
    fn = np.sum((pred == 0) & (y_test == 1))
    acc = (tp + tn) / len(y_test)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    results.append({'阈值': t, '准确率': acc, '精确率': prec, '召回率': rec, '空报': fp, '漏报': fn})
    print(f"  {t:.1f}  {acc*100:>7.1f}% {prec*100:>7.1f}% {rec*100:>7.1f}% {fp:>8} {fn:>8}")

# ====== 可视化：阈值vs指标曲线 ======
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：不同阈值下的表现
ax1 = axes[0]
ts = np.arange(0.1, 1.0, 0.05)
accs, precs, recs, fps, fns = [], [], [], [], []
true_total = y_test.sum()

for t in ts:
    pred = (prob >= t).astype(int)
    tp = np.sum((pred == 1) & (y_test == 1))
    tn = np.sum((pred == 0) & (y_test == 0))
    fp = np.sum((pred == 1) & (y_test == 0))
    fn = np.sum((pred == 0) & (y_test == 1))
    accs.append((tp+tn)/len(y_test))
    precs.append(tp/(tp+fp) if (tp+fp)>0 else 0)
    recs.append(tp/(tp+fn) if (tp+fn)>0 else 0)
    fps.append(fp)
    fns.append(fn)

ax1.plot(ts, precs, 'g-', linewidth=2, label='精确率（说暴雨→真暴雨）')
ax1.plot(ts, recs, 'r-', linewidth=2, label='召回率（真暴雨→抓住了）')
ax1.plot(ts, accs, 'b--', linewidth=1.5, label='准确率', alpha=0.5)
ax1.axvline(x=0.5, color='orange', linestyle=':', alpha=0.7, label='默认阈值0.5')
ax1.set_xlabel('阈值')
ax1.set_ylabel('指标值')
ax1.set_title('不同阈值下 精确率 vs 召回率 变化')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 右图：空报 vs 漏报
ax2 = axes[1]
ax2.plot(ts, fps, 'orange', linewidth=2, label='空报次数（虚惊一场）')
ax2.plot(ts, fns, 'red', linewidth=2, label='漏报次数（出大事了！）')
ax2.axvline(x=0.5, color='orange', linestyle=':', alpha=0.7)
ax2.set_xlabel('阈值')
ax2.set_ylabel('次数')
ax2.set_title('不同阈值下 空报 vs 漏报 权衡')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 标出几个关键点
for t_val in [0.3, 0.5, 0.7, 0.9]:
    idx = np.abs(ts - t_val).argmin()
    ax2.annotate(f'阈值={t_val:.1f}\n空报{fps[idx]}次\n漏报{fns[idx]}次',
                xy=(t_val, fps[idx]), xytext=(t_val+0.08, fps[idx]+10),
                fontsize=8, ha='center')

plt.suptitle('阈值调优 — 在漏报和空报之间找平衡', fontsize=15)
plt.tight_layout()
plt.savefig('8_8_阈值调优.png', dpi=150, bbox_inches='tight')
print("\n图片已保存: 8_8_阈值调优.png")
