#!/usr/bin/env python
"""01b_download_molecular.py - download molecular data used by scripts 13 and 14.

* TCGA lgggbm_tcga_pub discrete copy-number calls (data_cna.txt; GISTIC,
  -2 = homozygous deletion, 2 = high-level amplification) from the cBioPortal
  datahub LFS host. Used for CDKN2A/B homozygous deletion and EGFR
  amplification (refined WHO 2021 classification, TCGA only).
* TCGA PanCancer Atlas RNA-seq (RSEM, batch-corrected) for the LGG and GBM
  studies (data_mrna_seq_v2_rsem.txt), same patients as lgggbm_tcga_pub.
* CGGA mRNAseq_693 and mRNAseq_325 RSEM gene expression (release 20200506).

Checksums and provenance are appended to outputs/checksums/data_manifest.json.
"""
from __future__ import annotations

import hashlib
import io
import json
import pathlib
import ssl
import sys
import urllib.request
import zipfile
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.utils import get_logger, get_paths  # noqa: E402

log = get_logger("01b_download_molecular")
LFS = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/"
CGGA = "https://www.cgga.org.cn/download/20200506/"
UA = {"User-Agent": "Mozilla/5.0"}
_CTX = ssl.create_default_context()

TARGETS = [
    ("TCGA", "lgggbm_tcga_pub", LFS + "lgggbm_tcga_pub/data_cna.txt", "tcga/lgggbm_tcga_pub/data_cna.txt"),
    ("TCGA", "lgg_tcga_pan_can_atlas_2018", LFS + "lgg_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem.txt",
     "tcga/expression/lgg_pancan_data_mrna_seq_v2_rsem.txt"),
    ("TCGA", "gbm_tcga_pan_can_atlas_2018", LFS + "gbm_tcga_pan_can_atlas_2018/data_mrna_seq_v2_rsem.txt",
     "tcga/expression/gbm_pancan_data_mrna_seq_v2_rsem.txt"),
    ("CGGA", "mRNAseq_693", CGGA + "CGGA.mRNAseq_693.RSEM-genes.20200506.txt.zip",
     "cgga/expression/CGGA.mRNAseq_693.RSEM-genes.20200506.txt"),
    ("CGGA", "mRNAseq_325", CGGA + "CGGA.mRNAseq_325.RSEM-genes.20200506.txt.zip",
     "cgga/expression/CGGA.mRNAseq_325.RSEM-genes.20200506.txt"),
]


def _get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=600, context=_CTX) as r:
        return r.read()


def main() -> int:
    paths = get_paths()
    manifest_path = paths.checksums / "data_manifest.json"
    manifest = json.load(open(manifest_path)) if manifest_path.exists() else []
    now = datetime.now(timezone.utc).isoformat()
    for cohort, dataset, url, rel in TARGETS:
        dest = paths.data_raw / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        b = _get(url)
        if url.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(b)) as z:
                name = [n for n in z.namelist() if n.endswith(".txt")][0]
                b = z.read(name)
        dest.write_bytes(b)
        manifest = [m for m in manifest if not (m.get("dataset") == dataset and m.get("file") == dest.name)]
        manifest.append(dict(cohort=cohort, dataset=dataset, file=dest.name, bytes=len(b),
                             sha256=hashlib.sha256(b).hexdigest(), source=url, download_utc=now))
        log.info("%s %s (%d bytes)", dataset, dest.name, len(b))
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
