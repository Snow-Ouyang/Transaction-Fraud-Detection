"""Verify the exact original CSV bytes, including an actionable LFS-pointer check."""
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]


def verify():
    manifest = json.loads((ROOT / 'data/manifest.json').read_text(encoding='utf-8'))
    for item in manifest['files']:
        path = ROOT / item['path']
        if not path.is_file():
            raise FileNotFoundError(f'{path.name} missing; run git lfs pull')
        with path.open('rb') as stream:
            if stream.read(100).startswith(b'version https://git-lfs.github.com/spec/v1'):
                raise RuntimeError(f'{path.name} is an LFS pointer; run git lfs pull')
            stream.seek(0)
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if path.stat().st_size != item['bytes'] or digest != item['sha256']:
            raise RuntimeError(f'Original bytes do not match manifest: {path.name}')
        print(f'PASS {path.name}: {item["rows"]:,} rows, SHA-256 {digest}', flush=True)
    return manifest


if __name__ == '__main__':
    verify()
