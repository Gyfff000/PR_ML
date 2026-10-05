"""Download the public UCI Dry Bean dataset; no GitHub account is needed."""
import hashlib
import io
import json
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen
import zipfile

raw = Path(__file__).resolve().parent / "data/raw"
raw.mkdir(parents=True, exist_ok=True)
url = "https://archive.ics.uci.edu/static/public/602/dry%2Bbean%2Bdataset.zip"
target = raw / "Dry_Bean_Dataset.xlsx"
if target.exists():
    print("Dataset already exists; kept the existing file.")
else:
    with urlopen(Request(url, headers={"User-Agent": "Python course project"}), timeout=90) as response:
        payload = response.read()
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        member = next(n for n in archive.namelist() if n.endswith("Dry_Bean_Dataset.xlsx"))
        target.write_bytes(archive.read(member))
    (raw / "dry_bean.zip").write_bytes(payload)
    metadata = {"url": url, "dataset_page": "https://archive.ics.uci.edu/dataset/602/dry+bean+dataset",
                "doi": "10.24432/C50S4B", "license": "CC BY 4.0",
                "download_date": datetime.now().astimezone().isoformat(),
                "zip_sha256": hashlib.sha256(payload).hexdigest()}
    (raw / "source.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("Downloaded:", target)
