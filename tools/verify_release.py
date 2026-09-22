"""Verify released raw data, frozen results and artwork without API calls."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[1]
def main():
    manifest=json.loads((ROOT/'RELEASE_MANIFEST.json').read_text())
    failures=[]
    for entry in manifest['files']:
        path=ROOT/entry['path']
        if not path.is_file(): failures.append(entry['path']+' (missing)'); continue
        if hashlib.sha256(path.read_bytes()).hexdigest()!=entry['sha256']:
            failures.append(entry['path']+' (hash mismatch)')
    if failures: print('\n'.join(failures)); return 1
    print(f"Verified {len(manifest['files'])} data, artifact and figure files.")
    return 0
if __name__=='__main__': raise SystemExit(main())
