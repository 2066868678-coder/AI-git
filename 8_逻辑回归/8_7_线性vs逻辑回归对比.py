"""
对比：线性回归 vs 逻辑回归 — 暴雨预测
=================================
同样数据，两种方法，哪个更适合分类？
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request, json
from sklearn.linear_model import LinearRegression, LogisticRegression

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# ====== 加载北京数据 ======
url = ('https://archive-api.open-meteo.com/v1/archive?'
       'latitude=39.9&longitude=116.4'
       '&start_date=2015-01-01&end_date=2024-12-31'
       '&daily=precipitation_sum,temperature_2m_max,temperature_2m_min,'
       'relative_humidity_2m_mean,pressure_msl_mean,cloud_cover_mean,'
       'wind_speed_10m_max&timezone=Asia/Shanghai')
resp = urllib.request.urlopen(url, timeout=15)
data = json.loads(resp.read())
daily = data['daily']

df = pd.DataFrame({
    '降雨量': daily['precipitation_sum'],
    '最高温': daily['temperature_2m_max'],
    '最低温': daily['temperature_2m_min'],
    '湿度': daily['relative_humidity_2m_mean'],
    '气压': daily['pressure_msl_mean'],
    '云量': daily['cloud_cover_mean'],
    '风速': daily['wind_speed_10m_max'],
})
df = df.dropna()
df['暴雨'] = (df['降雨量'] >= 25).astype(int)

feature_cols = ['湿度', '最低温', '风速', '气压']
X = df[feature_cols].values
y = df['暴雨'].values

split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# ====== ① 线性回归硬做分类 ======
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_score = lr.predict(X_test)           # 输出连续值（不是概率！）
lr_pred = (lr_score >= 0.5).astype(int) # 强行套阈值

# ====== ② 逻辑回归 ======
logit = LogisticRegression(max_iter=2000, class_weight='balanced')
logit.fit(X_train, y_train)
logit_pred = logit.predict(X_test)
logit_prob = logit.predict_proba(X_test)[:, 1]

# ====== 对比输出 ======
def print_metrics(y_true, y_pred, name):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    print(f"\n{name}")
    print("-" * 30)
    print(f"准确率: {(tp+tn)/len(y_true)*100:.1f}%")
    print(f"精确率: {tp/(tp+fp)*100:.1f}%" if (tp+fp)>0 else "精确率: 无")
    print(f"召回率: {tp/(tp+fn)*100:.1f}%" if (tp+fn)>0 else "召回率: 无")
    print(f"预测暴雨 {tp+fp}次，其中真正暴雨{tp}次，空报{fp}次")
    print(f"漏报 {fn}次")
    return tp, tn, fp, fn

tp1, tn1, fp1, fn1 = print_metrics(y_test, lr_pred, "线性回归（强行分类）")
tp2, tn2, fp2, fn2 = print_metrics(y_test, logit_pred, "逻辑回归")

# ====== 可视化：四张图 ======
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 线性回归输出值分布（关键！）
ax1 = axes[0, 0]
ax1.scatter(range(len(lr_score)), lr_score, c=y_test, cmap='RdYlGn', alpha=0.5, s=12)
ax1.axhline(y=0.5, color='orange', linestyle='--', label='阈值0.5')
ax1.axhline(y=0, color='gray', linestyle=':')
ax1.axhline(y=1, color='gray', linestyle=':')
ax1.set_ylabel('输出值')
ax1.set_title('线性回归输出：有大量<0和>1的值\n不适合用于分类', fontsize=11)
ax1.legend()
ax1.grid(True, alpha=0.2)

# 2. 线性回归混淆矩阵
ax2 = axes[0, 1]
m1 = np.array([[tn1, fp1], [fn1, tp1]])
ax2.imshow(m1, cmap='Blues', alpha=0.7)
for i in range(2):
    for j in range(2):
        ax2.text(j, i, str(m1[i, j]), ha='center', va='center', fontsize=16, fontweight='bold')
ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
ax2.set_xticklabels(['预测非暴雨', '预测暴雨'])
ax2.set_yticklabels(['实际非暴雨', '实际暴雨'])
ax2.set_title('线性回归 混淆矩阵', fontsize=12)

# 3. 逻辑回归概率分布（关键！）
ax3 = axes[1, 0]
ax3.scatter(range(len(logit_prob)), logit_prob, c=y_test, cmap='RdYlGn', alpha=0.5, s=12)
ax3.axhline(y=0.5, color='orange', linestyle='--', label='阈值0.5')
ax3.set_ylabel('概率')
ax3.set_title('逻辑回归输出：严格在0~1之间\n天然适合做分类', fontsize=11)
ax3.legend()
ax3.grid(True, alpha=0.2)
ax3.set_ylim(-0.05, 1.05)

# 4. 逻辑回归混淆矩阵
ax4 = axes[1, 1]
m2 = np.array([[tn2, fp2], [fn2, tp2]])
ax4.imshow(m2, cmap='Blues', alpha=0.7)
for i in range(2):
    for j in range(2):
        ax4.text(j, i, str(m2[i, j]), ha='center', va='center', fontsize=16, fontweight='bold')
ax4.set_xticks([0, 1]); ax4.set_yticks([0, 1])
ax4.set_xticklabels(['预测非暴雨', '预测暴雨'])
ax4.set_yticklabels(['实际非暴雨', '实际暴雨'])
ax4.set_title('逻辑回归 混淆矩阵', fontsize=12)

plt.suptitle('线性回归 vs 逻辑回归 — 北京暴雨预测', fontsize=15)
plt.tight_layout()
plt.savefig('8_7_线性vs逻辑回归对比.png', dpi=150, bbox_inches='tight')
print("\n图片已保存: 8_7_线性vs逻辑回归对比.png")
