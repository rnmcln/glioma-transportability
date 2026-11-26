"""Render the five concordance figures with corrected group and time-origin labels."""
from importlib import import_module
from pathlib import Path
import sys
from types import SimpleNamespace

repo, target = map(Path,sys.argv[1:])
sys.path[:0]=[str(repo),str(repo/'scripts')]
from src.utils import get_paths

source=repo/'scripts/14_figures_tables.py'
code=source.read_text()
for old,new in (
    ('a1.set_xlim(0.64, 0.84)','a1.set_xlim(0.63, 0.84)'),
    ('Glioblastoma, IDH-wildtype','IDH-wildtype tumours'),
    ('Astrocytoma, IDH-mutant','IDH-mutant astrocytoma proxy'),
    ('ax.set_xlabel("Months from diagnosis")','ax.set_xlabel("Months from diagnosis" if coh == "TCGA_internal" else "Months from surgery")')
):
    assert old in code,old
    code=code.replace(old,new)
namespace={'__name__':'reviewed_concordance_figures','__file__':str(source)}
exec(compile(code,str(source),'exec'),namespace)
p=get_paths()
paths=SimpleNamespace(**{attr:getattr(p,attr) for attr in ('root','data_raw','data_processed','models','tables','logs','checksums')},figures=target)
namespace['set_style']()
for graphic in ('fig_decomposition','fig_within','fig_calibration_main','figS_assoc','figS_km'):
    namespace[graphic](paths)
