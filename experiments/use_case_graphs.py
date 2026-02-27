import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec

# ── Data ─────────────────────────────────────────────────────────────────────
participants = ['P1', 'P2', 'P3', 'P4', 'P5']
question_labels = [
    'Ease of finding information (Q4.1)',
    'Clarity of search steps (Q4.2)',
    'Interface intuitive (Q5.1)',
    'Information understandable (Q6.1)',
    'Results clear & interpretable (Q6.2)',
    'Loading speed (Q8.1)',
]

scores = np.array([
    [4, 5, 4, 5, 3], [5, 5, 5, 5, 4], [4, 5, 5, 4, 3],
    [4, 5, 5, 5, 4], [4, 5, 4, 4, 4], [5, 3, 5, 5, 4],
])
nps_scores = [8, 7, 10, 9, 7]

# ── Style ─────────────────────────────────────────────────────────────────────
BASE_SIZE = 15
LEGEND_SIZE = 13
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': BASE_SIZE,
    'axes.titlesize': BASE_SIZE + 1,
    'pdf.fonttype': 42,
})
GRAY = '#888888'

# ── Layout: Forzamos alineación vertical ──────────────────────────────────────
# Ajustamos figsize y usamos márgenes fijos en GridSpec para alinear alturas
fig = plt.figure(figsize=(10, 3.8)) 
gs = GridSpec(1, 2, figure=fig, width_ratios=[1.8, 1.4], 
              wspace=0.8, left=0.1, right=0.95, top=0.85, bottom=0.15)

ax_heat = fig.add_subplot(gs[0, 0])
ax_bar  = fig.add_subplot(gs[0, 1])

# ── (a) Heatmap ───────────────────────────────────────────────────────────────
# Usamos aspect='auto' para que rellene el espacio definido por el GridSpec
im = ax_heat.imshow(scores, aspect='auto', cmap='RdYlGn', vmin=1, vmax=5)

ax_heat.set_xticks(range(5))
ax_heat.set_xticklabels(participants)
ax_heat.set_yticks(range(len(question_labels)))
ax_heat.set_yticklabels(question_labels)
ax_heat.tick_params(left=False, bottom=False)

for (i, j), val in np.ndenumerate(scores):
    ax_heat.text(j, i, str(val), ha='center', va='center',
                 fontweight='bold', color='white' if val <= 2 else '#222222')

cb = fig.colorbar(im, ax=ax_heat, fraction=0.04, pad=0.04, ticks=[1, 3, 5])
cb.outline.set_visible(False)
ax_heat.set_title('(a) Individual Scores (Likert: 1-5)', pad=15)

# ── (b) NPS Barplot ──────────────────────────────────────────────────────────
y_pos = np.arange(len(participants))
colors = ['#55A868' if s >= 9 else (GRAY if s >= 7 else '#C44E52') for s in nps_scores]

# Aumentamos ligeramente el height para que visualmente llenen el mismo alto
bars = ax_bar.barh(y_pos, nps_scores, color=colors, height=0.5, edgecolor='white', linewidth=0.5)

for bar in bars:
    width = bar.get_width()
    ax_bar.text(width + 0.2, bar.get_y() + bar.get_height()/2, 
                f'{width}', va='center', fontweight='bold', fontsize=BASE_SIZE)

ax_bar.set_yticks(y_pos)
ax_bar.set_yticklabels(participants)
# Ajustamos el ylim para que el área de dibujo sea idéntica a la del heatmap (0 a N-1)
ax_bar.set_ylim(-0.5, len(participants) - 0.5) 
ax_bar.set_xlim(0, 11)
ax_bar.set_xticks([0, 5, 7, 9, 10])
ax_bar.spines[['top', 'right']].set_visible(False)

# ── Leyenda Vertical ──────────────────────────────────────────────────────────
patches = [
    mpatches.Patch(color='#55A868', label='Promoter'),
    mpatches.Patch(color=GRAY, label='Passive'),
    mpatches.Patch(color='#C44E52', label='Detractor')
]

ax_bar.legend(handles=patches, loc='center left', bbox_to_anchor=(-0.75, 0.5),
              ncol=1, frameon=True, handlelength=1.2, labelspacing=1.2, 
              fontsize=LEGEND_SIZE)

ax_bar.set_title('(b) Recommendation (NPS: 0-10)', pad=15)

# ── Save ──────────────────────────────────────────────────────────────────────
plt.savefig('user_study_aligned.pdf', bbox_inches='tight', dpi=300)
print("Gráfico alineado guardado como user_study_aligned.pdf")