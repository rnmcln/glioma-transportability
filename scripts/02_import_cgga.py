#!/usr/bin/env python
"""02_import_cgga.py - download and unpack CGGA clinical files.

Source: CGGA portal (https://www.cgga.org.cn/download.jsp). The download servlet
requires a dataset-specific `type` parameter; the session is primed by first
requesting the download page. Files are saved zipped and unzipped, with SHA256
checksums and provenance recorded in the data manifest.

If the portal is unreachable, place the clinical .txt files manually under
data_raw/cgga/<release>/ and re-run; the importer will use whatever is present.
"""
from __future__ import annotations

import hashlib
import io
import json
import pathlib
import ssl
import sys
import urllib.parse
import urllib.request
import zipfile
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src.utils import get_logger, get_paths  # noqa: E402

log = get_logger("02_import_cgga")
BASE = "https://www.cgga.org.cn/download"
PAGE = "https://www.cgga.org.cn/download.jsp"
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124 Safari/537.36"
RELEASES = {
    "mRNAseq_693": ("download/20200506/CGGA.mRNAseq_693_clinical.20200506.txt.zip", "mRNAseq_693_clinical", "20200506"),
    "mRNAseq_325": ("download/20200506/CGGA.mRNAseq_325_clinical.20200506.txt.zip", "mRNAseq_325_clinical", "20200506"),
    "mRNA_array_301": ("download/20200506/CGGA.mRNA_array_301_clinical.20200506.txt.zip", "mRNA_array_301_clinical", "20200506"),
    "WEseq_286": ("download/20200506/CGGA.WEseq_286_clinical.20200506.txt.zip", "WEseq_286_clinical", "20200506"),
}
_CTX = ssl.create_default_context()


def main() -> int:
    paths = get_paths()
    manifest_path = paths.checksums / "data_manifest.json"
    manifest = json.load(open(manifest_path)) if manifest_path.exists() else []
    manifest = [m for m in manifest if m.get("cohort") != "CGGA"]
    now = datetime.now(timezone.utc).isoformat()

    opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor())
    opener.addheaders = [("User-Agent", UA), ("Referer", PAGE)]
    try:
        opener.open(PAGE, timeout=60).read()  # prime session cookie
    except Exception as exc:
        log.warning("could not reach CGGA portal (%s); will use any local files", exc)

    for rel, (fp, typ, tm) in RELEASES.items():
        outdir = paths.cgga_raw / rel
        outdir.mkdir(parents=True, exist_ok=True)
        url = f"{BASE}?" + urllib.parse.urlencode({"file": fp, "type": typ, "time": tm})
        try:
            b = opener.open(url, timeout=120).read()
            assert b[:2] == b"PK", "not a zip payload"
            (outdir / pathlib.Path(fp).name).write_bytes(b)
            manifest.append(dict(cohort="CGGA", dataset=rel, file=pathlib.Path(fp).name,
                                 bytes=len(b), sha256=hashlib.sha256(b).hexdigest(),
                                 source=url, download_utc=now))
            with zipfile.ZipFile(io.BytesIO(b)) as z:
                for n in z.namelist():
                    if n.endswith(".txt") and not n.startswith("__MACOSX"):
                        data = z.read(n)
                        (outdir / pathlib.Path(n).name).write_bytes(data)
                        manifest.append(dict(cohort="CGGA", dataset=rel, file=pathlib.Path(n).name,
                                             bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                                             source=url + " (unzipped)", download_utc=now))
            log.info("CGGA %s downloaded", rel)
        except Exception as exc:
            local = list(outdir.glob("*clinical*.txt"))
            if local:
                log.info("CGGA %s: using local file %s", rel, local[0].name)
            else:
                log.warning("CGGA %s unavailable and no local file present: %s", rel, exc)

    json.dump(manifest, open(manifest_path, "w"), indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
