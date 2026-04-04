import json, sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimSun', 'WenQuanYi Micro Hei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, 'tools')
from train_side_state_model import extract_features, compute_dyn_stationary, STATE_TO_ID

ID_TO_STATE = {v: k for k, v in STATE_TO_ID.items()}

# Load data
data = json.loads(Path('data/nuscenes/annotations/states_user_mini_val.json').read_text())
X_all, y_all = [], []
for tr in data.get('tracks', []):
    state = tr.get('state', 'going_straight')
    if state not in STATE_TO_ID:
        continue
    X_all.append(extract_features(tr, use_radar=True))
    y_all.append(STATE_TO_ID[state])
X_all = np.stack(X_all)
y_all = np.array(y_all, dtype=np.int64)

STATE_NAMES_CN = {
    'going_straight': '直行',
    'going_left': '向左',
    'going_right': '向右',
    'parking': '停车',
}
CLASSES = [STATE_NAMES_CN[ID_TO_STATE[i]] for i in range(4)]

outdir = Path('output/visualizations')
outdir.mkdir(parents=True, exist_ok=True)

# ── 1. Confusion Matrix Heatmap ──────────────────────────────────────────────
print("Generating confusion matrix...")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
all_preds, all_true = [], []
for tr_idx, te_idx in skf.split(X_all, y_all):
    Xtr, Xte = X_all[tr_idx], X_all[te_idx]
    ytr, yte = y_all[tr_idx], y_all[te_idx]
    f_mean, f_std = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-6
    clf = RandomForestClassifier(
        n_estimators=400, max_depth=8,
        class_weight={0: 1, 1: 1, 2: 1, 3: 0.3},
        random_state=42, n_jobs=-1)
    clf.fit((Xtr - f_mean) / f_std, ytr)
    all_preds.extend(clf.predict((Xte - f_mean) / f_std).tolist())
    all_true.extend(yte.tolist())

all_true = np.array(all_true)
all_preds = np.array(all_preds)
cm = confusion_matrix(all_true, all_preds, labels=list(range(4)))

# 行归一化：每行除以该行总数，显示百分比
cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

fig, ax = plt.subplots(figsize=(7.5, 6.5))
# 单元格中显示：百分比 + 原始数量
annot = np.empty_like(cm_norm, dtype=object)
for i in range(4):
    for j in range(4):
        annot[i, j] = f'{cm_norm[i, j]:.1f}%\n({cm[i, j]})'

sns.heatmap(cm_norm, annot=annot, fmt='', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES,
            ax=ax, annot_kws={'size': 13}, cbar_kws={'label': '占该类真实样本的比例 (%)'},
            linewidths=0.5, linecolor='white', vmin=0, vmax=100)
ax.set_xlabel('预测类别', fontsize=14)
ax.set_ylabel('真实类别', fontsize=14)
ax.set_title('混淆矩阵 (5折交叉验证, n=808)', fontsize=15, fontweight='bold', pad=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12, rotation=0)
# 在对角线格子加粗边框
for i in range(4):
    ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                edgecolor='darkblue', linewidth=2.5))
plt.tight_layout()
plt.savefig(outdir / 'confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> confusion_matrix.png")

# ── 2. Per-class Metrics Bar Chart ──────────────────────────────────────────
print("Generating per-class metrics...")
prec = []
recall = []
for i in range(4):
    mask = all_true == i
    tp = ((all_preds == i) & mask).sum()
    fp = ((all_preds == i) & ~mask).sum()
    fn = (mask & (all_preds != i)).sum()
    prec.append(tp / (tp + fp) if tp + fp > 0 else 0)
    recall.append(tp / (tp + fn) if tp + fn > 0 else 0)

f1_scores = [2 * p * r / (p + r) if p + r > 0 else 0 for p, r in zip(prec, recall)]

x = np.arange(4)
width = 0.25

fig, ax = plt.subplots(figsize=(9, 5.5))
bars_p = ax.bar(x - width, prec, width, label='精确率', color='#3498db', alpha=0.85, edgecolor='white')
bars_r = ax.bar(x,       recall, width, label='召回率', color='#e74c3c', alpha=0.85, edgecolor='white')
bars_f = ax.bar(x + width, f1_scores, width, label='F1分数', color='#2ecc71', alpha=0.85, edgecolor='white')

for bars in [bars_p, bars_r, bars_f]:
    for bar in bars:
        h = bar.get_height()
        if h > 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f'{h:.1%}',
                    ha='center', va='bottom', fontsize=10)

ax.set_ylabel('分数', fontsize=12)
ax.set_title('各状态类别性能指标 (5折交叉验证)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(CLASSES, fontsize=12)
ax.set_ylim(0, 1.18)
ax.legend(fontsize=11, loc='upper right')
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(outdir / 'per_class_metrics.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> per_class_metrics.png")

# ── 3. Accuracy Progression Across Optimization Stages ─────────────────────────
print("Generating accuracy progression...")
stages = [
    'Baseline\n(4-class)',
    'Simplified\nStates',
    'Extended\nRadar Feats',
    'Two-Stage\nClassifier',
    'Data\nCleaning',
    'Hyperparam\nOptimization',
]
stages_cn = ['初始基线\n(4分类)', '简化状态\n定义', '扩展雷达\n特征',
              '两阶段\n分类器', '数据清洗\n(删除误标)', '超参优化\n(park_w=0.3)']
accs = [0.583, 0.670, 0.690, 0.715, 0.792, 0.845]
stds = [0.021, 0.020, 0.018, 0.020, 0.009, 0.005]
colors = ['#bdc3c7', '#3498db', '#3498db', '#9b59b6', '#e67e22', '#27ae60']

fig, ax = plt.subplots(figsize=(10, 5.5))
bars = ax.bar(range(len(stages)), accs, color=colors, alpha=0.85, edgecolor='white', width=0.55)
for bar, acc, std in zip(bars, accs, stds):
    ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02,
            f'{acc:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.axhline(y=0.80, color='red', linestyle='--', alpha=0.6, linewidth=1.5, label='80%基线')
ax.axhline(y=0.845, color='green', linestyle='--', alpha=0.6, linewidth=1.5, label='最终84.5%')
ax.set_ylabel('准确率', fontsize=13)
ax.set_title('优化各阶段准确率变化', fontsize=15, fontweight='bold')
ax.set_xticks(range(len(stages_cn)))
ax.set_xticklabels(stages_cn, fontsize=10)
ax.set_ylim(0, 1.0)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(outdir / 'accuracy_progression.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> accuracy_progression.png")

# ── 4. dyn_stationary Distribution by Class ───────────────────────────────────
print("Generating dyn_stationary distribution...")
dyn_by_class = {i: [] for i in range(4)}
for tr in data.get('tracks', []):
    state = tr.get('state', 'going_straight')
    if state not in STATE_TO_ID:
        continue
    sid = STATE_TO_ID[state]
    dyn = compute_dyn_stationary(tr)
    dyn_by_class[sid].append(dyn)

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes[0]
parts = ax.violinplot([dyn_by_class[i] for i in range(4)],
                      positions=range(4), showmeans=True, showmedians=True)
for pc in parts['bodies']:
    pc.set_alpha(0.7)
parts['cmeans'].set_color('red')
parts['cmedians'].set_color('orange')
ax.set_xticks(range(4))
ax.set_xticklabels(CLASSES, fontsize=12)
ax.set_ylabel('dyn_stationary 值', fontsize=12)
ax.set_title('dyn_stationary 分布（小提琴图）', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0.4, color='red', linestyle='--', alpha=0.6, linewidth=1.5, label='阈值0.4')
ax.legend(fontsize=10)

ax = axes[1]
colors_box = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
bp = ax.boxplot([dyn_by_class[i] for i in range(4)],
                positions=range(4), patch_artist=True, widths=0.5)
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xticks(range(4))
ax.set_xticklabels(CLASSES, fontsize=12)
ax.set_ylabel('dyn_stationary 值', fontsize=12)
ax.set_title('dyn_stationary 分布（箱线图）', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0.4, color='red', linestyle='--', alpha=0.6, linewidth=1.5, label='阈值0.4')
ax.legend(fontsize=10)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.suptitle('各状态类别的 dyn_stationary 特征分布\n（停车检测关键特征）', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(outdir / 'dyn_stationary_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> dyn_stationary_distribution.png")

# ── 5. Combined Summary Figure ──────────────────────────────────────────────
print("Generating combined summary figure...")
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

# CM - row normalized
ax1 = fig.add_subplot(gs[0, 0])
annot_s = np.empty_like(cm_norm, dtype=object)
for i in range(4):
    for j in range(4):
        annot_s[i, j] = f'{cm_norm[i, j]:.1f}%'
sns.heatmap(cm_norm, annot=annot_s, fmt='', cmap='Blues', xticklabels=CLASSES,
            yticklabels=CLASSES, ax=ax1, annot_kws={'size': 12},
            linewidths=0.5, linecolor='white', cbar_kws={'shrink': 0.8},
            vmin=0, vmax=100)
ax1.set_xlabel('预测类别', fontsize=12)
ax1.set_ylabel('真实类别', fontsize=12)
ax1.set_title('混淆矩阵（行归一化%）', fontsize=13, fontweight='bold')
for i in range(4):
    ax1.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='darkblue', linewidth=2.5))

# Per-class metrics
ax2 = fig.add_subplot(gs[0, 1])
bars_p = ax2.bar(x - width, prec, width, label='精确率', color='#3498db', alpha=0.85)
bars_r = ax2.bar(x, recall, width, label='召回率', color='#e74c3c', alpha=0.85)
bars_f = ax2.bar(x + width, f1_scores, width, label='F1分数', color='#2ecc71', alpha=0.85)
for bars in [bars_p, bars_r, bars_f]:
    for bar in bars:
        h = bar.get_height()
        if h > 0.05:
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f'{h:.0%}',
                     ha='center', va='bottom', fontsize=9)
ax2.set_ylabel('分数', fontsize=11)
ax2.set_title('各状态类别性能指标', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(CLASSES, fontsize=11)
ax2.set_ylim(0, 1.18)
ax2.legend(fontsize=10)
ax2.grid(axis='y', alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# Progression
ax3 = fig.add_subplot(gs[1, 0])
bars3 = ax3.bar(range(len(stages)), accs, color=colors, alpha=0.85, edgecolor='white', width=0.5)
for bar, acc in zip(bars3, accs):
    ax3.text(bar.get_x() + bar.get_width() / 2, acc + 0.015,
             f'{acc:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax3.set_ylabel('准确率', fontsize=11)
ax3.set_title('优化各阶段准确率变化', fontsize=13, fontweight='bold')
ax3.set_xticks(range(len(stages_cn)))
ax3.set_xticklabels(stages_cn, fontsize=9)
ax3.set_ylim(0, 1.0)
ax3.grid(axis='y', alpha=0.3)
ax3.axhline(y=0.80, color='red', linestyle='--', alpha=0.5, label='80%基线')
ax3.legend(fontsize=10)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Dyn distribution
ax4 = fig.add_subplot(gs[1, 1])
bp = ax4.boxplot([dyn_by_class[i] for i in range(4)],
                 positions=range(4), patch_artist=True, widths=0.5)
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax4.set_xticks(range(4))
ax4.set_xticklabels(CLASSES, fontsize=11)
ax4.set_ylabel('dyn_stationary', fontsize=11)
ax4.set_title('dyn_stationary 分布（停车检测关键特征）', fontsize=13, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)
ax4.axhline(y=0.4, color='red', linestyle='--', alpha=0.6, linewidth=1.5, label='阈值0.4')
ax4.legend(fontsize=10)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

fig.suptitle('CenterFusion 行为分类优化结果总览', fontsize=16, fontweight='bold', y=0.99)
plt.savefig(outdir / 'summary_figure.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> summary_figure.png")

# ── 6. Data Distribution Pie + Bar ─────────────────────────────────────────
print("Generating data distribution figure...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))

# Pie chart
ax = axes[0]
dist = [187, 76, 58, 487]
pie_colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
wedges, texts, autotexts = ax.pie(dist, labels=CLASSES, autopct='%1.1f%%',
                                    colors=pie_colors, explode=(0, 0, 0, 0.05),
                                    shadow=False, startangle=90)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight('bold')
ax.set_title('数据分布（清洗后, n=808）', fontsize=12, fontweight='bold')

# Before vs After bar
ax = axes[1]
before = [187, 76, 58, 760]
after  = [187, 76, 58, 487]
x = np.arange(4)
w = 0.35
b1 = ax.bar(x - w/2, before, w, label='清洗前 (n=1081)', color='#bdc3c7', alpha=0.85)
b2 = ax.bar(x + w/2, after, w, label='清洗后 (n=808)', color='#27ae60', alpha=0.85)
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 5,
                    f'{int(h)}', ha='center', va='bottom', fontsize=9)
ax.set_ylabel('样本数量', fontsize=12)
ax.set_title('清洗前后样本数量对比', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(CLASSES, fontsize=11)
ax.legend(fontsize=11)
ax.grid(axis='y', alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(outdir / 'data_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> data_distribution.png")

print(f"\nAll visualizations saved to: {outdir}")
import os
for f in sorted(os.listdir(outdir)):
    fpath = outdir / f
    print(f"  {f} ({os.path.getsize(fpath)/1024:.1f} KB)")
