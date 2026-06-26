#!/usr/bin/env python
"""01_download_tcga.py - download TCGA lgggbm_tcga_pub clinical files.

Source: cBioPortal datahub (https://github.com/cBioPortal/datahub). The clinical
tables are stored with git-LFS, so they are fetched from the LFS media host;
small meta files fall back to the raw host. SHA256 checksums and provenance are
recorded in outputs/checksums/data_manifest.json.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import ssl
import sys
import urllib.request
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.utils import get_logger, get_paths  # noqa: E402

log = get_logger("01_download_tcga")
LFS = "https://media.githubusercontent.com/media/cBioPortal/datahub/master/public/lgggbm_tcga_pub/"
RAW = "https://raw.githubusercontent.com/cBioPortal/datahub/master/public/lgggbm_tcga_pub/"
FILES = ["data_clinical_patient.txt", "data_clinical_sample.txt", "meta_study.txt",
         "meta_clinical_patient.txt", "meta_clinical_sample.txt"]
UA = {"User-Agent": "Mozilla/5.0"}
_CTX = ssl.create_default_context()


def _get(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120, context=_CTX) as r:
        return r.read()


def main() -> int:
    paths = get_paths()
    paths.tcga_raw.mkdir(parents=True, exist_ok=True)
    manifest_path = paths.checksums / "data_manifest.json"
    manifest = json.load(open(manifest_path)) if manifest_path.exists() else []
    manifest = [m for m in manifest if m.get("cohort") != "TCGA"]
    now = datetime.now(timezone.utc).isoformat()
    for f in FILES:
        try:
            b = _get(LFS + f)
            src = LFS + f
        except Exception:
            b = _get(RAW + f)
            src = RAW + f
        (paths.tcga_raw / f).write_bytes(b)
        manifest.append(dict(cohort="TCGA", dataset="lgggbm_tcga_pub", file=f,
                             bytes=len(b), sha256=hashlib.sha256(b).hexdigest(),
                             source=src, download_utc=now))
        log.info("TCGA %s (%d bytes)", f, len(b))
    json.dump(manifest, open(manifest_path, "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
