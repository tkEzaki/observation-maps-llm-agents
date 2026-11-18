"""Decompress raw journals that are stored gzip-compressed and check their SHA-256."""
from pathlib import Path
import gzip, hashlib, json
ROOT = Path(__file__).resolve().parents[1]
def main():
    table = json.loads((ROOT/'independent_studies/LARGE_FILES.json').read_text())
    for packed, info in table.items():
        data = gzip.decompress((ROOT/packed).read_bytes())
        assert len(data) == info['restored_bytes'] and hashlib.sha256(data).hexdigest() == info['restored_sha256'], packed
        (ROOT/info['restored_path']).write_bytes(data)
        print('restored', info['restored_path'])
    return 0
if __name__ == '__main__': raise SystemExit(main())
