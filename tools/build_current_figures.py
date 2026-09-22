"""Offline current figures: new main/S27 plots plus frozen-vector SI casing."""
from pathlib import Path
import subprocess,sys,shutil
ROOT=Path(__file__).resolve().parents[1]
def main():
    subprocess.run([sys.executable,'-B','tools/restore_frozen_results.py'],cwd=ROOT,check=True)
    for module in ['uppercase_figure_panels_20260922','uppercase_figure_references_20260922',
                   'build_figures_v4','build_main_figures_v4','build_figS27_v4']:
        subprocess.run([sys.executable,'-B','-m','analysis.publication.'+module],cwd=ROOT,check=True)
    output=ROOT/'figures/publication'
    output.mkdir(exist_ok=True)
    for folder in ['panel_conversion','rendered_main']:
        for source in (ROOT/'figures'/folder).glob('*.pdf'):
            shutil.copy2(source,output/source.name)
    assert len(list(output.glob('*.pdf')))==44
    print('Current figures: figures/publication (44 PDFs).')
if __name__=='__main__': main()
