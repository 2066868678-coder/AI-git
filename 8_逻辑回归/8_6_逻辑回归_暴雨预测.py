"""
逻辑回归 — 北京暴雨预测（真实气象数据）
==================================
数据源：Open-Meteo API（欧洲中期天气预报中心）
北京2015-2024年逐日气象数据
特征：最高温、最低温、湿度、气压、云量、风速
标签：0=非暴雨, 1=暴雨（日降雨量≥25mm，中国暴雨标准）
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request, json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
plt.rcParams['axes.unicode_minus'] = False

# ====== ① 从 API 拉北京真实气象数据 ======
print("正在下载北京10年气象数据...")
url = ('https://archive-api.open-meteo.com/v1/archive?'
       'latitude=39.9&longitude=116.4'
       '&start_date=2015-01-01&end_date=2024-12-31'
       '&daily=precipitation_sum,temperature_2m_max,temperature_2m_min,'
       'relative_humidity_2m_mean,pressure_msl_mean,cloud_cover_mean,'
       'wind_speed_10m_max&timezone=Asia/Shanghai')
resp = urllib.request.urlopen(url, timeout=15)
data = json.loads(resp.read())
daily = data['daily']

# 转成 DataFrame
df = pd.DataFrame({
    'date': daily['time'],
    '降雨量': daily['precipitation_sum'],
    '最高温': daily['temperature_2m_max'],
    '最低温': daily['temperature_2m_min'],
    '湿度': daily['relative_humidity_2m_mean'],
    '气压': daily['pressure_msl_mean'],
    '云量': daily['cloud_cover_mean'],
    '风速': daily['wind_speed_10m_max'],
})
df = df.dropna()  # 去掉空值

# 暴雨标签：日降雨量≥25mm（中国气象局暴雨标准）
df['暴雨'] = (df['降雨量'] >= 25).astype(int)

print(f"\n总记录: {len(df)} 天（2015-2024年）")
print(f"暴雨天数: {df['暴雨'].sum()} 天 ({df['暴雨'].mean()*100:.1f}%)")
print(f"最大日降雨量: {df['降雨量'].max():.1f} mm")

# ====== ② 构造特征和标签 ======
feature_cols = ['最高温', '最低温', '湿度', '气压', '云量', '风速']
X = df[feature_cols].values
y = df['暴雨'].values

# 按时间分割：前8年训练，后2年测试（时间序列不能用随机分割）
split = int(len(df) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"\n训练集: {len(X_train)} 天 (2015-2022)")
print(f"测试集: {len(X_test)} 天 (2023-2024)")

# ====== ③ 训练逻辑回归 ======
model = LogisticRegression(max_iter=2000, class_weight='balanced')
model.fit(X_train, y_train)

# 看模型学到的各特征权重
print(f"\n各特征对暴雨的影响（权重w）：")
for name, w in zip(feature_cols, model.coef_[0]):
    print(f"  {name}: {w:+.4f}  → {'湿度越大越暴雨' if w>0 else '值越小越暴雨'}")
print(f"偏置 b: {model.intercept_[0]:.4f}")

# ====== ④ 测试集评估 ======
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)

# 混淆矩阵
tp = np.sum((y_pred == 1) & (y_test == 1))
tn = np.sum((y_pred == 0) & (y_test == 0))
fp = np.sum((y_pred == 1) & (y_test == 0))
fn = np.sum((y_pred == 0) & (y_test == 1))

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n{'='*40}")
print(f"测试集结果（2023-2024年，共{len(X_test)}天）")
print(f"{'='*40}")
print(f"真实暴雨: {y_test.sum()} 天, 非暴雨: {len(y_test)-y_test.sum()} 天")
print(f"准确率: {accuracy*100:.1f}%")
print(f"精确率: {precision*100:.1f}%  （模型说暴雨→真暴雨的比例）")
print(f"召回率: {recall*100:.1f}%  （真暴雨→模型抓住了的比例）")
print(f"F1分数: {f1:.3f}")

# ====== ⑤ 可视化 ======
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 图1：混淆矩阵
ax1 = axes[0]
matrix = np.array([[tn, fp], [fn, tp]])
ax1.imshow(matrix, cmap='Blues', alpha=0.7)
for i in range(2):
    for j in range(2):
        ax1.text(j, i, str(matrix[i, j]) + f'\n({matrix[i,j]/len(y_test)*100:.1f}%)',
                ha='center', va='center', fontsize=14, fontweight='bold')
ax1.set_xticks([0, 1])
ax1.set_yticks([0, 1])
ax1.set_xticklabels(['预测非暴雨', '预测暴雨'])
ax1.set_yticklabels(['实际非暴雨', '实际暴雨'])
ax1.set_xlabel('预测结果')
ax1.set_ylabel('真实情况')
ax1.set_title(f'混淆矩阵 — 北京暴雨预测\n准确率={accuracy*100:.1f}%')

# 图2：特征重要性（系数大小）
ax2 = axes[1]
weights = model.coef_[0]
colors = ['red' if w > 0 else 'blue' for w in weights]
ax2.barh(feature_cols, weights, color=colors, alpha=0.7)
ax2.axvline(x=0, color='black', linewidth=0.5)
ax2.set_xlabel('权重大小（正=暴雨概率↑，负=暴雨概率↓）')
ax2.set_title('各特征对暴雨的影响程度')
ax2.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('8_6_北京暴雨预测.png', dpi=150, bbox_inches='tight')
print("\n图片已保存: 8_6_北京暴雨预测.png")
