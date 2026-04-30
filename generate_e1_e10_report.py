"""
Generate E1–E10 comparison charts (AUC + ACC) for weekly report.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties
import numpy as np

FONT_PATH = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
fp = FontProperties(fname=FONT_PATH, size=11)
fp_title = FontProperties(fname=FONT_PATH, size=13, weight='bold')
fp_small = FontProperties(fname=FONT_PATH, size=9)
fp_legend = FontProperties(fname=FONT_PATH, size=10)

plt.rcParams['axes.unicode_minus'] = False

# ── 完整實驗數據 ──────────────────────────────────────────────────
EXPERIMENTS = [
    'E1\nBalanced',
    'E2\nFeat.Align',
    'E3-M1\nDANN(NC_AD)',
    'E3-M2\nDANN(All)',
    'E4\nTask-λ',
    'E5\nHybrid-λ★',
    'E6\nCond.Dual',
    'E6v3\nNC-only',
    'E7\nMTL Joint',
    'E8\nMTL+Adapt',
    'E9\nMTL+Ord',
    'E10\nMTL+Boost',
]

AUC = {
    'NC vs AD':  [0.7574, 0.7836, 0.7846, 0.7846, 0.7846, 0.7846,
                  0.7774, 0.7752, 0.7649, 0.7701, 0.7917, 0.7868],
    'NC vs MCI': [0.6940, 0.6716, 0.7209, 0.7293, 0.7170, 0.7293,
                  0.7223, 0.7293, 0.6791, 0.6985, 0.6801, 0.7033],
    'MCI vs AD': [0.7782, 0.7670, 0.7656, 0.7634, 0.7748, 0.7748,
                  0.7604, 0.7656, 0.7853, 0.7963, 0.8003, 0.8042],
}

ACC = {
    'NC vs AD':  [0.6928, 0.7255, 0.7451, 0.7451, 0.7451, 0.7451,
                  0.6993, 0.7190, 0.7386, 0.6797, 0.7124, 0.7320],
    'NC vs MCI': [0.6771, 0.6823, 0.6979, 0.6667, 0.6771, 0.6667,
                  0.6875, 0.6667, 0.6146, 0.6146, 0.6302, 0.6302],
    'MCI vs AD': [0.7410, 0.7194, 0.7194, 0.6906, 0.7770, 0.7770,
                  0.7194, 0.7194, 0.7554, 0.7050, 0.7194, 0.7122],
}

N = len(EXPERIMENTS)
x = np.arange(N)
COLORS = {'NC vs AD': '#4C72B0', 'NC vs MCI': '#DD8452', 'MCI vs AD': '#55A868'}
# E5 (idx=5) and E9/E10 are key milestones → highlight background
MILESTONES = {5: '#fff9c4', 11: '#e8f5e9', 10: '#fce4ec'}

def add_milestone_bg(ax, indices_colors):
    ylim = ax.get_ylim()
    for idx, color in indices_colors.items():
        ax.axvspan(idx - 0.5, idx + 0.5, alpha=0.25, color=color, zorder=0)

# ── Figure 1: AUC 趨勢折線圖 ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 6))
for task, vals in AUC.items():
    ax.plot(x, vals, 'o-', color=COLORS[task], linewidth=2.5, markersize=7, label=task)
    for i, v in enumerate(vals):
        ax.annotate(f'{v:.3f}', (i, v), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=7.5, color=COLORS[task])

# milestone backgrounds
for idx, color in MILESTONES.items():
    ax.axvspan(idx - 0.5, idx + 0.5, alpha=0.2, color=color, zorder=0)

ax.axhline(0.729, color=COLORS['NC vs MCI'], linestyle='--', alpha=0.4, linewidth=1.2)
ax.axhline(0.785, color=COLORS['NC vs AD'],  linestyle='--', alpha=0.4, linewidth=1.2)

ax.set_xticks(x)
ax.set_xticklabels(EXPERIMENTS, fontproperties=fp_small)
ax.set_ylabel('AUC', fontproperties=fp)
ax.set_ylim(0.62, 0.87)
ax.set_title('E1–E10 AUC 趨勢比較（5-fold CV × 3 seeds OOF）', fontproperties=fp_title)
ax.legend(prop=fp_legend, loc='lower right')
ax.grid(axis='y', alpha=0.3)

# 標注重要里程碑
for label, xi, yi, dy in [
    ('E5 Best\nSingle', 5, 0.7846, 0.025),
    ('E9 NC_AD\nMCI_AD↑', 10, 0.7917, 0.025),
    ('E10\nMCI_AD 新高', 11, 0.8042, -0.055),
]:
    ax.annotate(label, xy=(xi, yi), xytext=(xi, yi + dy),
                ha='center', fontsize=8, color='#333',
                arrowprops=dict(arrowstyle='->', color='#555', lw=1.2),
                fontproperties=fp_small)

fig.tight_layout()
fig.savefig('/home/wei-chi/Data/script/results/E1_to_E10_auc_trend.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved: E1_to_E10_auc_trend.png')

# ── Figure 2: ACC 趨勢折線圖 ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 6))
for task, vals in ACC.items():
    ax.plot(x, vals, 's--', color=COLORS[task], linewidth=2.2, markersize=7, label=task)
    for i, v in enumerate(vals):
        ax.annotate(f'{v:.3f}', (i, v), textcoords='offset points',
                    xytext=(0, 8), ha='center', fontsize=7.5, color=COLORS[task])

for idx, color in MILESTONES.items():
    ax.axvspan(idx - 0.5, idx + 0.5, alpha=0.2, color=color, zorder=0)

ax.set_xticks(x)
ax.set_xticklabels(EXPERIMENTS, fontproperties=fp_small)
ax.set_ylabel('ACC', fontproperties=fp)
ax.set_ylim(0.57, 0.83)
ax.set_title('E1–E10 ACC 趨勢比較（5-fold CV × 3 seeds OOF）', fontproperties=fp_title)
ax.legend(prop=fp_legend, loc='lower right')
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig('/home/wei-chi/Data/script/results/E1_to_E10_acc_trend.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved: E1_to_E10_acc_trend.png')

# ── Figure 3: 分組 AUC 長條圖（全部實驗）──────────────────────────
fig, ax = plt.subplots(figsize=(18, 7))
width = 0.25
tasks = list(AUC.keys())
for ti, task in enumerate(tasks):
    offset = (ti - 1) * width
    bars = ax.bar(x + offset, AUC[task], width, label=task, color=COLORS[task], alpha=0.85)

# E5 highlight box
for xi in [5]:
    ax.axvspan(xi - 0.5, xi + 0.5, alpha=0.12, color='gold', zorder=0)
for xi in [10, 11]:
    ax.axvspan(xi - 0.5, xi + 0.5, alpha=0.12, color='lightgreen', zorder=0)

ax.set_xticks(x)
ax.set_xticklabels(EXPERIMENTS, fontproperties=fp_small)
ax.set_ylabel('AUC', fontproperties=fp)
ax.set_ylim(0.60, 0.87)
ax.set_title('E1–E10 AUC 分組長條圖（每組三個任務）', fontproperties=fp_title)
ax.legend(prop=fp_legend)
ax.grid(axis='y', alpha=0.3)
ax.axhline(0.7, color='gray', linestyle=':', alpha=0.5)
ax.axhline(0.8, color='gray', linestyle=':', alpha=0.5)
fig.tight_layout()
fig.savefig('/home/wei-chi/Data/script/results/E1_to_E10_auc_bar.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved: E1_to_E10_auc_bar.png')

# ── Figure 4: 分組 ACC 長條圖 ────────────────────────────────────
fig, ax = plt.subplots(figsize=(18, 7))
for ti, task in enumerate(tasks):
    offset = (ti - 1) * width
    ax.bar(x + offset, ACC[task], width, label=task, color=COLORS[task], alpha=0.85)

for xi in [5]:
    ax.axvspan(xi - 0.5, xi + 0.5, alpha=0.12, color='gold', zorder=0)
for xi in [10, 11]:
    ax.axvspan(xi - 0.5, xi + 0.5, alpha=0.12, color='lightgreen', zorder=0)

ax.set_xticks(x)
ax.set_xticklabels(EXPERIMENTS, fontproperties=fp_small)
ax.set_ylabel('ACC', fontproperties=fp)
ax.set_ylim(0.55, 0.82)
ax.set_title('E1–E10 ACC 分組長條圖（每組三個任務）', fontproperties=fp_title)
ax.legend(prop=fp_legend)
ax.grid(axis='y', alpha=0.3)
ax.axhline(0.7, color='gray', linestyle=':', alpha=0.5)
fig.tight_layout()
fig.savefig('/home/wei-chi/Data/script/results/E1_to_E10_acc_bar.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved: E1_to_E10_acc_bar.png')

# ── Figure 5: 每個 Task 獨立趨勢（3×2 subplot）──────────────────
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
metrics = [('AUC', AUC), ('ACC', ACC)]
for row, (metric_name, metric_data) in enumerate(metrics):
    for col, task in enumerate(tasks):
        ax = axes[row][col]
        vals = metric_data[task]
        colors_bar = ['#c62828' if i == np.argmax(vals) else COLORS[task] for i in range(N)]
        bars = ax.bar(x, vals, color=colors_bar, alpha=0.8, edgecolor='white', linewidth=0.5)
        for i, (bar, v) in enumerate(zip(bars, vals)):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.003, f'{v:.3f}',
                    ha='center', va='bottom', fontsize=7, fontproperties=fp_small)
        ax.set_xticks(x)
        ax.set_xticklabels(EXPERIMENTS, rotation=45, ha='right', fontproperties=fp_small, fontsize=7)
        ylim_lo = min(vals) - 0.03
        ylim_hi = max(vals) + 0.04
        ax.set_ylim(ylim_lo, ylim_hi)
        ax.set_title(f'{task} — {metric_name}', fontproperties=fp_title)
        ax.grid(axis='y', alpha=0.3)
        # best value annotation
        best_idx = int(np.argmax(vals))
        ax.annotate('Best', xy=(best_idx, vals[best_idx]),
                    xytext=(best_idx, vals[best_idx] + 0.02),
                    ha='center', fontsize=8, color='#c62828',
                    fontproperties=fp_small)

fig.suptitle('E1–E10 各任務 AUC / ACC 逐實驗比較（紅色=最高）', fontproperties=fp_title, fontsize=14, y=1.01)
fig.tight_layout()
fig.savefig('/home/wei-chi/Data/script/results/E1_to_E10_per_task.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print('Saved: E1_to_E10_per_task.png')

print('\n所有圖表已生成完畢。')
