"""Generate error analysis charts for the paper."""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sns.set_theme(style="whitegrid", font_scale=0.95)

# ── Chart 1: Average subscores (grouped bar) ──────────────────

data1 = pd.DataFrame({
    'Configuration': ['Gem. Pro\n+ Reas.'] * 4
                    + ['Gem. Flash\n+ Scr.'] * 4
                    + ['Mist. Med.\n+ Reas.'] * 4
                    + ['Mist. Sm.\n+ Scr.'] * 4,
    'Component': ['$S_h$', '$S_s$', '$S_a$', '$S_f$'] * 4,
    'Score': [0.99, 0.89, 0.85, 0.97,
              1.00, 0.99, 0.98, 1.00,
              0.87, 0.74, 0.38, 0.41,
              0.95, 0.91, 0.82, 0.87],
})

palette = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']

fig, ax = plt.subplots(figsize=(6, 3.5))
sns.barplot(
    data=data1, x='Configuration', y='Score',
    hue='Component', palette=palette, ax=ax,
)
ax.set_ylim(0, 1.15)
ax.set_ylabel('Average Score')
ax.set_xlabel('')
ax.legend(title='', loc='upper right', fontsize=9)

for container in ax.containers:
    for bar in container:
        h = bar.get_height()
        if h < 0.95:
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.02,
                f'{h:.2f}', ha='center', va='bottom', fontsize=7,
            )

plt.tight_layout()
plt.savefig('subscores_chart.png', dpi=300, bbox_inches='tight')
print('Saved subscores_chart.png')
plt.close()


# ── Chart 2: Root cause (stacked bar) ─────────────────────────

configs = [
    'Gem. Pro\n+ Reas.',
    'Gem. Flash\n+ Scr.',
    'Mist. Med.\n+ Reas.',
    'Mist. Sm.\n+ Scr.',
]

data2 = pd.DataFrame({
    'Configuration': configs,
    'Perfect':       [165, 195,  36, 159],
    '$S_h$ root':    [ 14,   0,  79,  40],
    '$S_s$ root':    [ 21,   5,  36,   1],
    '$S_a$ root':    [  0,   0,  25,   0],
    '$S_f$ root':    [  0,   0,  24,   0],
})

colors = ['#4CAF50', '#2196F3', '#FF9800', '#F44336', '#9C27B0']
categories = ['Perfect', '$S_h$ root', '$S_s$ root', '$S_a$ root', '$S_f$ root']

fig2, ax2 = plt.subplots(figsize=(6, 3.5))
x = np.arange(len(configs))
bottom = np.zeros(len(configs))

for cat, color in zip(categories, colors):
    vals = data2[cat].values
    ax2.bar(x, vals, bottom=bottom, label=cat, color=color, width=0.6)
    bottom += vals

ax2.set_ylabel('Number of Cases')
ax2.set_xticks(x)
ax2.set_xticklabels(configs, fontsize=9)
ax2.set_ylim(0, 220)
ax2.legend(loc='upper right', fontsize=8)
sns.despine(left=True)

plt.tight_layout()
plt.savefig('rootcause_chart.png', dpi=300, bbox_inches='tight')
print('Saved rootcause_chart.png')
plt.close()
