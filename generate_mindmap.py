"""生成 AI 学习路线思维导图 (支持中文)"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# ── 设置中文字体 ──
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# ── 画布设置 ──
fig, ax = plt.subplots(figsize=(22, 14))
ax.set_xlim(0, 22)
ax.set_ylim(0, 14)
ax.axis('off')
fig.patch.set_facecolor('#FAFAFA')

# ── 颜色方案 ──
DONE    = '#5CB85C'  # 已完成 - 绿色
DONE_BG = '#DFF0D8'
CURR    = '#F0AD4E'  # 当前 - 橙色
CURR_BG = '#FCF8E3'
FUTURE  = '#5BC0DE'  # 未学 - 蓝色
FUTURE_BG='#D9EDF7'
CORE    = '#D9534F'  # 核心 - 红色
CORE_BG = '#F2DEDE'

# ── 辅助函数 ──
def filled_box(x, y, w, h, text, bg, fg='white', fs=11, bold=True, style='round,pad=0.15'):
    """画实色圆角矩形"""
    b = FancyBboxPatch((x, y), w, h, boxstyle=style,
                       facecolor=bg, edgecolor=bg, linewidth=0)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs,
            fontweight='bold' if bold else 'normal', color=fg)

def border_box(x, y, w, h, text, bg, border='#888', fs=10, fg='#333'):
    """画浅色带边框盒子"""
    b = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08',
                       facecolor=bg, edgecolor=border, linewidth=1.5)
    ax.add_patch(b)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=fs,
            fontweight='normal', color=fg)

# ═══════════════════════════════════════════════════════
# 数据
# ═══════════════════════════════════════════════════════
# 每个站：(名称, 标题颜色, 背景色, 前缀, [(课程名, 边框颜色, 背景色)])
STATIONS = [
    ("第1站 * 回归基础", DONE, DONE_BG, '[完]', [
        ("一元线性回归", DONE, DONE_BG),
        ("回归评估指标", DONE, DONE_BG),
        ("多元线性回归", DONE, DONE_BG),
        ("多项式回归", DONE, DONE_BG),
        ("[此] 正则化 Ridge/Lasso", CURR, CURR_BG),
    ]),
    ("第2站 * 分类算法", FUTURE, FUTURE_BG, '[待]', [
        ("逻辑回归(二分类)", FUTURE, FUTURE_BG),
        ("KNN最近邻", FUTURE, FUTURE_BG),
        ("决策树", FUTURE, FUTURE_BG),
        ("分类模型评估", FUTURE, FUTURE_BG),
    ]),
    ("第3站 * 集成学习", FUTURE, FUTURE_BG, '[待]', [
        ("随机森林", FUTURE, FUTURE_BG),
        ("XGBoost / LightGBM", FUTURE, FUTURE_BG),
    ]),
    ("第4站 * 无监督学习", FUTURE, FUTURE_BG, '[待]', [
        ("K-Means聚类", FUTURE, FUTURE_BG),
        ("PCA降维", FUTURE, FUTURE_BG),
    ]),
    ("第5站 * 深度学习", FUTURE, FUTURE_BG, '[待]', [
        ("神经网络MLP", FUTURE, FUTURE_BG),
        ("PyTorch入门", FUTURE, FUTURE_BG),
        ("分类实战", FUTURE, FUTURE_BG),
    ]),
    ("[火] 第6站 * 时序预测 ***", CORE, CORE_BG, '[火]', [
        ("RNN循环神经网络", CORE, CORE_BG),
        ("[星] LSTM (水利核心!)", CORE, CORE_BG),
        ("水文预测实战项目", CORE, CORE_BG),
    ]),
    ("第7站 * 计算机视觉", FUTURE, FUTURE_BG, '[待]', [
        ("CNN卷积神经网络", FUTURE, FUTURE_BG),
        ("遥感图像分割", FUTURE, FUTURE_BG),
    ]),
    ("第8站 * NLP大模型", FUTURE, FUTURE_BG, '[待]', [
        ("词嵌入与文本分类", FUTURE, FUTURE_BG),
        ("LLM应用 + RAG", FUTURE, FUTURE_BG),
    ]),
    ("第9站 * 强化学习", FUTURE, FUTURE_BG, '[待]', [
        ("强化学习基础", FUTURE, FUTURE_BG),
        ("[杯] 综合实战项目", FUTURE, FUTURE_BG),
    ]),
]

# ═══════════════════════════════════════════════════════
# 布局
# ═══════════════════════════════════════════════════════
ROOT_X,  ROOT_W  = 0.3,   3.0
ROOT_Y,  ROOT_H  = 6.2,   1.0
ST_X,    ST_W    = 4.0,   3.2
ST_GAP_Y         = 1.22
ST_TOP           = 12.8
COURSE_X         = 7.6
COURSE_W         = 2.3
COURSE_GAP_X     = 2.65

# ═══════════════════════════════════════════════════════
# 绘制标题
# ═══════════════════════════════════════════════════════
ax.text(11, 13.6, "AI 学习路线 * 思维导图", ha='center', va='center',
        fontsize=20, fontweight='bold', color='#2C3E50')
ax.text(11, 13.0, "已完成 6/23 课 (26%)  |  每周学2次 约 3-4 个月学完  |  下一站: 正则化 Ridge/Lasso",
        ha='center', va='center', fontsize=10, color='#777')

# ═══════════════════════════════════════════════════════
# 绘制根节点
# ═══════════════════════════════════════════════════════
filled_box(ROOT_X, ROOT_Y, ROOT_W, ROOT_H, '水文AI 知识体系',
           '#2C3E50', 'white', fs=18)
ax.text(ROOT_X + ROOT_W/2, ROOT_Y - 0.35,
        '回归 -> 分类 -> 集成 -> 深度学习 -> 时序预测(LSTM) -> 拓展',
        ha='center', va='center', fontsize=8, color='#999', fontstyle='italic')

# ═══════════════════════════════════════════════════════
# 绘制各站
# ═══════════════════════════════════════════════════════
ROOT_RIGHT = ROOT_X + ROOT_W

for i, (name, hdr_bg, _, prefix, courses) in enumerate(STATIONS):
    y_center = ST_TOP - i * ST_GAP_Y
    st_y = y_center - 0.28
    st_h = 0.55

    # 连线：根 → 标题
    rx = ROOT_X + ROOT_W
    ry = ROOT_Y + ROOT_H/2
    ax.plot([rx, ST_X - 0.05], [ry, st_y + st_h/2],
            color='#BBB', lw=1.5, zorder=0)

    # 标题框
    filled_box(ST_X, st_y, ST_W, st_h, name, hdr_bg, fg='#333', fs=10)

    # 课程
    for j, (cname, cborder, cbg) in enumerate(courses):
        cx = COURSE_X + j * COURSE_GAP_X
        cy = st_y + 0.03
        cw = COURSE_W
        ch = 0.45

        # 箭头
        if j == 0:
            ax.annotate('', xy=(cx, cy + ch/2),
                        xytext=(ST_X + ST_W + 0.08, cy + ch/2),
                        arrowprops=dict(arrowstyle='->', color='#AAA', lw=1.3))
        else:
            px = COURSE_X + (j-1) * COURSE_GAP_X + COURSE_W
            ax.annotate('', xy=(cx, cy + ch/2),
                        xytext=(px + 0.08, cy + ch/2),
                        arrowprops=dict(arrowstyle='->', color='#BBB', lw=1.0))

        # 文字颜色
        if cborder == DONE:
            tcolor = '#1a5e1a'
        elif cborder == CURR:
            tcolor = '#8a6d00'
        elif cborder == CORE:
            tcolor = '#8a1a1a'
        else:
            tcolor = '#1a5276'

        border_box(cx, cy, cw, ch, cname, cbg, border=cborder, fg=tcolor, fs=9)

# ═══════════════════════════════════════════════════════
# 图例
# ═══════════════════════════════════════════════════════
leg_items = [
    (DONE_BG, DONE, '[完] 已完成'),
    (CURR_BG, CURR, '[此] 当前学习'),
    (FUTURE_BG, FUTURE, '[待] 即将学习'),
    (CORE_BG, CORE, '[火] 核心重点'),
]
for k, (bg, border, label) in enumerate(leg_items):
    lx = 0.3 + k * 5.0
    ly = 0.25
    b = FancyBboxPatch((lx, ly), 0.55, 0.35, boxstyle='round,pad=0.05',
                       facecolor=bg, edgecolor=border, linewidth=2)
    ax.add_patch(b)
    ax.text(lx + 0.7, ly + 0.17, label, ha='left', va='center', fontsize=9, color='#555')

# ═══════════════════════════════════════════════════════
# 保存
# ═══════════════════════════════════════════════════════
plt.savefig('AI学习路线_思维导图.png', dpi=200, bbox_inches='tight',
            facecolor=fig.get_facecolor(), edgecolor='none')
plt.close()
print("OK: AI学习路线_思维导图.png")
