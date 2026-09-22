"""Fig. S27 (Claude action distributions at kappa = 12) in the shared journal style.

Run from the repository root:  python -m analysis.sciadv_revision.build_figS27_v4

Same data as integrate_claude_figures.supplement(). Changes: the action colours of
Figs. 2B and 5E (magenta retard, grey stay, navy advance; the previous green advance
was close to the intervals colour), Arial-metric type, bold capital panel letters,
and no in-figure title (the caption carries it). Output: figures/sciadv_v4.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from analysis.sciadv_revision.build_figures import REPS, NAMES
from analysis.sciadv_revision.integrate_claude_figures import SOURCES, BLOCKS
from analysis.sciadv_revision.build_figures_v4 import S, COLORS, panel_title, save, setup_v4

ACTIONS = [('p_retard', S.ACTION['p_minus'], 'retard'),
           ('p_stay', S.ACTION['p_zero'], 'stay'),
           ('p_advance', S.ACTION['p_plus'], 'advance')]


def build():
    setup_v4()
    curves = pd.read_csv(SOURCES[1]/'offset_action_curves.csv')
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.4), sharey=True)
    fig.subplots_adjust(left=.085, right=.985, bottom=.17, top=.9, wspace=.2, hspace=.62)
    for row, (block, _, _) in enumerate(BLOCKS):
        for col, rep in enumerate(REPS):
            ax = axes[row, col]
            sub = curves[(curves.block == block) & (curves.representation == rep) & (curves.concentration == 12)].sort_values('offset_radians')
            assert len(sub) == 36
            bottom = np.zeros(36)
            for field, color, _ in ACTIONS:
                ax.bar(sub.offset_radians, sub[field], width=.115, bottom=bottom, color=color, edgecolor='none')
                bottom += sub[field].to_numpy()
            assert np.allclose(bottom, 1.0)
            ax.axvline(0, color='white', lw=.6)
            ax.set(ylim=(0, 1), xlim=(-np.pi, np.pi), xticks=[-np.pi, 0, np.pi], xticklabels=['−π', '0', 'π'], yticks=[0, .5, 1],
                   xlabel=r'Rotation $\delta$')
            if col == 0:
                ax.set_ylabel('Action probability')
            panel_title(ax, 'ABCDEF'[row*3+col], f'{NAMES[col]}, block {row+1}')
            ax.title.set_color(COLORS[col])
    handles = [Line2D([], [], marker='s', ls='', ms=6, color=c, label=l) for _, c, l in ACTIONS]
    fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, fontsize=S.FS_SMALL, bbox_to_anchor=(.5, .0))
    save(fig, 'figS27_claude_actions')


if __name__ == '__main__':
    build()
