"""Figure 1 of the collective-forecast study.

Renders the matched-population comparison with the builder in analysis/publication/build_figures_v4.py
(same data and layout as the earlier release). Presentation changes only: bold lower-case panel
letters, the spelling 'centres' in labels and 'descriptions' in figure notes. Prompt lines printed
in panel a are the literal strings the models received.

Run from the repository root after tools/restore_frozen_results.py:
  python -B collective_forecast/build_fig1.py
Output: figures/collective_forecast/fig1_state_descriptions.pdf
"""
import os, sys
from pathlib import Path
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analysis.publication.build_figures_v4 as F

FIG = ROOT/'figures/collective_forecast'; FIG.mkdir(parents=True, exist_ok=True)


def save(fig, name):
    assert name.startswith('fig1'), name
    # The text uses 'description' for the state encoding. The panel-a note is also
    # raised slightly so that it clears the return arrow of the shared engine.
    import matplotlib.text as mtext
    hits = [t for t in fig.findobj(mtext.Text) if 'for all three encodings' in t.get_text()]
    assert len(hits) == 1, len(hits)
    x, y = hits[0].get_position(); hits[0].set_position((x, y + 0.05))
    for t in fig.findobj(mtext.Text):
        if 'encodings' in t.get_text():
            t.set_text(t.get_text().replace('encodings', 'descriptions'))
    fig.savefig(FIG/'fig1_state_descriptions.pdf')
    fig.savefig(FIG/'fig1_state_descriptions.png', dpi=300)
    plt.close(fig)


def label_at(fig, x_in, y_in, letter, text=''):
    W, H = fig.get_size_inches()
    fig.text(x_in/W, 1-y_in/H, letter.lower(), fontsize=F.FS_LETTER, fontweight='bold', ha='left', va='baseline', color=F.S.INK)
    if text:
        fig.text((x_in+F.S.LETTER_ADVANCE_MM*F.S.MM)/W, 1-y_in/H, text, fontsize=F.S.FS_TITLE, ha='left', va='baseline',
                 color='#2B2B2B')


if __name__ == '__main__':
    F.setup_v4()
    ep, rows, seeds, series = F.load()
    F.save = save
    F.label_at = label_at
    F.NAMES = ['moments', 'centres', 'intervals']
    F.fig1(seeds, series, ep)
    print('Figure 1 written to', FIG)
