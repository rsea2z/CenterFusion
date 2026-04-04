import json, sys, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
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

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASSES, yticklabels=CLASSES,
            ax=ax, annot_kws={'size': 14}, cbar_kws={'label': '样本数'},
            linewidths=0.5, linecolor='white')
ax.set_xlabel('Predicted Class', fontsize=13)
ax.set_ylabel('True Class', fontsize=13)
ax.set_title('Confusion Matrix (5-fold CV, n=808)', fontsize=14, fontweight='bold')
plt.xticks(fontsize=11)
plt.yticks(fontsize=11, rotation=0)
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
bars_p = ax.bar(x - width, prec, width, label='Precision', color='#3498db', alpha=0.85, edgecolor='white')
bars_r = ax.bar(x,       recall, width, label='Recall', color='#e74c3c', alpha=0.85, edgecolor='white')
bars_f = ax.bar(x + width, f1_scores, width, label='F1 Score', color='#2ecc71', alpha=0.85, edgecolor='white')

for bars in [bars_p, bars_r, bars_f]:
    for bar in bars:
        h = bar.get_height()
        if h > 0.05:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f'{h:.1%}',
                    ha='center', va='bottom', fontsize=9)

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Per-Class Performance Metrics (5-fold CV)', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{CLASSES[i]}\n({STATE_NAMES_CN[ID_TO_STATE[i]]})' for i in range(4)], fontsize=10)
ax.set_ylim(0, 1.14)
ax.legend(fontsize=10, loc='upper right')
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

fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar(range(len(stages)), accs, color=colors, alpha=0.85, edgecolor='white', width=0.55)
for bar, acc, std in zip(bars, accs, stds):
    ax.text(bar.get_x() + bar.get_width() / 2, acc + 0.02,
            f'{acc:.1%}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax.axhline(y=0.80, color='red', linestyle='--', alpha=0.6, linewidth=1.2, label='80% baseline')
ax.axhline(y=0.845, color='green', linestyle='--', alpha=0.6, linewidth=1.2, label='Final 84.5%')
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Accuracy Across Optimization Stages', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(stages_cn)))
ax.set_xticklabels(stages_cn, fontsize=9)
ax.set_ylim(0, 1.0)
ax.legend(fontsize=10)
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

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
parts = ax.violinplot([dyn_by_class[i] for i in range(4)],
                      positions=range(4), showmeans=True, showmedians=True)
for pc in parts['bodies']:
    pc.set_alpha(0.7)
parts['cmeans'].set_color('red')
parts['cmedians'].set_color('orange')
ax.set_xticks(range(4))
ax.set_xticklabels(CLASSES, fontsize=11)
ax.set_ylabel('dyn_stationary Value', fontsize=11)
ax.set_title('dyn_stationary Distribution (Violin)', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0.4, color='red', linestyle='--', alpha=0.6, label='Threshold=0.4')
ax.legend(fontsize=9)

ax = axes[1]
colors_box = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
bp = ax.boxplot([dyn_by_class[i] for i in range(4)],
                positions=range(4), patch_artist=True, widths=0.5)
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_xticks(range(4))
ax.set_xticklabels(CLASSES, fontsize=11)
ax.set_ylabel('dyn_stationary Value', fontsize=11)
ax.set_title('dyn_stationary Distribution (Box)', fontsize=12, fontweight='bold')
ax.grid(axis='y', alpha=0.3)
ax.axhline(y=0.4, color='red', linestyle='--', alpha=0.6, label='Threshold=0.4')
ax.legend(fontsize=9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.suptitle('dyn_stationary Distribution by Class\n(Key Feature for Parking Detection)', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig(outdir / 'dyn_stationary_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> dyn_stationary_distribution.png")

# ── 5. Combined Summary Figure ──────────────────────────────────────────────
print("Generating combined summary figure...")
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(16, 12))
gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30)

ax1 = fig.add_subplot(gs[0, 0])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES,
            yticklabels=CLASSES, ax=ax1, annot_kws={'size': 13},
            linewidths=0.5, linecolor='white', cbar_kws={'shrink': 0.8})
ax1.set_xlabel('Predicted', fontsize=11)
ax1.set_ylabel('True', fontsize=11)
ax1.set_title('Confusion Matrix', fontsize=12, fontweight='bold')

ax2 = fig.add_subplot(gs[0, 1])
bars_p = ax2.bar(x - width, prec, width, label='Precision', color='#3498db', alpha=0.85)
bars_r = ax2.bar(x, recall, width, label='Recall', color='#e74c3c', alpha=0.85)
bars_f = ax2.bar(x + width, f1_scores, width, label='F1', color='#2ecc71', alpha=0.85)
for bars in [bars_p, bars_r, bars_f]:
    for bar in bars:
        h = bar.get_height()
        if h > 0.05:
            ax2.text(bar.get_x() + bar.get_width() / 2, h + 0.01, f'{h:.0%}',
                     ha='center', va='bottom', fontsize=8)
ax2.set_ylabel('Score', fontsize=11)
ax2.set_title('Per-Class Metrics', fontsize=12, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(CLASSES, fontsize=10)
ax2.set_ylim(0, 1.15)
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

ax3 = fig.add_subplot(gs[1, 0])
bars3 = ax3.bar(range(len(stages)), accs, color=colors, alpha=0.85, edgecolor='white', width=0.5)
for bar, acc in zip(bars3, accs):
    ax3.text(bar.get_x() + bar.get_width() / 2, acc + 0.015,
             f'{acc:.1%}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax3.set_ylabel('Accuracy', fontsize=11)
ax3.set_title('Accuracy Progression', fontsize=12, fontweight='bold')
ax3.set_xticks(range(len(stages_cn)))
ax3.set_xticklabels(stages_cn, fontsize=8)
ax3.set_ylim(0, 1.0)
ax3.grid(axis='y', alpha=0.3)
ax3.axhline(y=0.80, color='red', linestyle='--', alpha=0.5)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

ax4 = fig.add_subplot(gs[1, 1])
bp = ax4.boxplot([dyn_by_class[i] for i in range(4)],
                 positions=range(4), patch_artist=True, widths=0.5)
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax4.set_xticks(range(4))
ax4.set_xticklabels(CLASSES, fontsize=10)
ax4.set_ylabel('dyn_stationary', fontsize=11)
ax4.set_title('dyn_stationary Distribution (Key Parking Feature)', fontsize=12, fontweight='bold')
ax4.grid(axis='y', alpha=0.3)
ax4.axhline(y=0.4, color='red', linestyle='--', alpha=0.6, label='Threshold 0.4')
ax4.legend(fontsize=9)
ax4.spines['top'].set_visible(False)
ax4.spines['right'].set_visible(False)

fig.suptitle('CenterFusion Behavior Classification Optimization Results', fontsize=15, fontweight='bold', y=0.98)
plt.savefig(outdir / 'summary_figure.png', dpi=150, bbox_inches='tight')
plt.close()
print("  -> summary_figure.png")

# ── 6. Data Distribution Pie + Bar ─────────────────────────────────────────
print("Generating data distribution figure...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

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
ax.set_title('Class Distribution After Cleaning (n=808)', fontsize=12, fontweight='bold')

# Before vs After bar
ax = axes[1]
before = [187, 76, 58, 760]
after  = [187, 76, 58, 487]
x = np.arange(4)
w = 0.35
b1 = ax.bar(x - w/2, before, w, label='Before Cleaning (n=1081)', color='#bdc3c7', alpha=0.85)
b2 = ax.bar(x + w/2, after, w, label='After Cleaning (n=808)', color='#27ae60', alpha=0.85)
for bars in [b1, b2]:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 5,
                    f'{int(h)}', ha='center', va='bottom', fontsize=8)
ax.set_ylabel('Number of Samples', fontsize=11)
ax.set_title('Sample Count: Before vs After Cleaning', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(CLASSES, fontsize=10)
ax.legend(fontsize=10)
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
