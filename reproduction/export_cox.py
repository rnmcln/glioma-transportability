"""Export the TCGA-CDR Cox coefficients and scaling metadata."""
import csv
from pathlib import Path
import sys
repo,out=map(Path,sys.argv[1:])
sys.path.insert(0,str(repo))
from src.registry import load_models
from src.utils import get_paths
keys=['B:cox','B:coxnet','R:cox','RA:cox','RAG:cox']
models=load_models(get_paths(),keys)
with out.open('w',newline='') as f:
    w=csv.writer(f)
    w.writerow(['model','term','log_hazard_coefficient','training_mean','training_sd','training_median','elastic_net_penalty'])
    for key,model in models.items():
        a=model.estimator
        e=model.encoder
        alpha=a.alphas_[0] if key=='B:coxnet' else ''
        for term,beta in zip(e.columns_,a.coef_.reshape(-1),strict=True):
            w.writerow([key,term,beta,e.num_mean_.get(term,''),e.num_std_.get(term,''),e.num_median_.get(term,''),alpha])
