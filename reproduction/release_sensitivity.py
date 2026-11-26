"""Pair decomposition after restricting comparisons to the same CGGA release."""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd

repo, out = map(Path, sys.argv[1:])
sys.path[:0] = [str(repo), str(repo / 'scripts')]
from src.utils import get_paths

df, predictions = importlib.import_module('10_subgroups_decomposition')._load(get_paths(), 'pooled_seq')
by_release = importlib.import_module('08_external_validation').cohorts(get_paths())
membership = {patient: name for name in ('mRNAseq_693','mRNAseq_325') for patient in by_release[name].patient_id}
score = predictions['B:coxnet'][0]
if score.ndim != 1:
    raise ValueError('Expected one frozen risk score per patient')
rows = []
for same_release in (False, True):
    for known_idh in (False, True):
        use = (df.idh_status != 'unknown').to_numpy() if known_idh else np.ones(len(df), bool)
        patient = df.loc[use]
        time, event = patient.os_time_days.to_numpy(), patient.os_event.to_numpy(bool)
        group = patient.integrated_class_harmonised.to_numpy()
        release = patient.patient_id.map(membership).to_numpy(str)
        p = (time[:, None] < time) | ((time[:, None] == time) & ~event)
        p &= event[:, None]
        if same_release:
            p &= release[:, None] == release
        delta = score[use, None] - score[use]
        value = np.where(abs(delta) <= 1e-8, .5, (delta > 0).astype(float))
        within = group[:, None] == group
        mats = np.stack([p & within, p & ~within, (p & within) * value, (p & ~within) * value]).astype(float)
        rng = np.random.default_rng(20261125)
        w = np.zeros((len(patient), 1001))
        w[:, 0] = 1
        for name in np.unique(release):
            i = np.where(release == name)[0]
            w[i, 1:] = rng.multinomial(len(i), np.repeat(1 / len(i), len(i)), size=1000).T
        nw, nb, sw, sb = (w * (mats @ w)).sum(axis=1)
        overall = (sw + sb) / (nw + nb)
        cwithin = sw / nw
        share = (sb - nb / 2) / (sw + sb - (nb + nw) / 2)
        res = dict(same_release=same_release,known_idh=known_idh,patients=len(patient),pairs=int(nw[0]+nb[0]))
        for key, x in [('c_all',overall),('c_within',cwithin),('attrib_between',share)]:
            res[key],res[key+'_lo'],res[key+'_hi']=x[0],*np.quantile(x[1:],(.025,.975))
        rows.append(res)
assert abs(rows[0]['c_all'] - .7759719363759168) < 1e-12
pd.DataFrame(rows).to_csv(out,index=False)
