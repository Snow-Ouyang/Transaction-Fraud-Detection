"""Read-only audit of staged Git contents, LFS data pointers, links and frozen sources."""
from pathlib import Path
import ast
import hashlib
import json
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def git(*args, input=None):
    return subprocess.run(['git', *args], cwd=ROOT, input=input, capture_output=True, check=True).stdout


def main():
    files = git('ls-files', '-z').decode('utf-8').strip('\0').split('\0')
    indexed = set(files)
    checks = {}
    forbidden_ext = {'.html', '.png', '.jpg', '.jpeg', '.svg', '.pdf', '.pkl', '.cbm', '.ubj', '.joblib'}
    checks['no_archives_graphics_models_vendored_dependencies'] = all(
        not (p.startswith(('archive/', 'tmp/', 'outputs/')) or '/.deps/' in p or
             Path(p).suffix in forbidden_ext or Path(p).name.startswith('report_zh.')) for p in files)
    pointer_hashes = {}
    for item in json.loads((ROOT/'data/manifest.json').read_text())['files']:
        blob = git('show', ':'+item['path']).decode('utf-8')
        expected = f'version https://git-lfs.github.com/spec/v1\noid sha256:{item["sha256"]}\nsize {item["bytes"]}\n'
        checks[item['path']+'_exact_lfs_pointer'] = blob == expected
        pointer_hashes[item['path']] = item['sha256']
    py_files = [p for p in files if p.endswith('.py')]
    for p in py_files:
        ast.parse(git('show', ':'+p).decode('utf-8-sig'), filename=p)
    checks['python_sources_parse'] = True
    json_files = [p for p in files if p.endswith('.json')]
    for p in json_files:
        json.loads(git('show', ':'+p).decode('utf-8-sig'))
    checks['all_json_parse'] = True
    broken = []
    for p in [f for f in files if f.endswith('.md')]:
        text = git('show', ':'+p).decode('utf-8-sig')
        for target in re.findall(r'(?<!!)\[[^\]]+\]\(([^)]+)\)', text):
            target = target.strip('<>').split('#')[0]
            if not target or re.match(r'^[a-z]+:', target):
                continue
            resolved = (ROOT/Path(p).parent/target).resolve()
            try:
                rel = resolved.relative_to(ROOT).as_posix()
            except ValueError:
                broken.append((p, target)); continue
            if rel not in indexed and not any(f.startswith(rel.rstrip('/')+'/') for f in files):
                broken.append((p, target))
    checks['markdown_links_resolve_in_tracked_content'] = not broken
    manifest = json.loads((ROOT/'models/leave_one_out/manifest.json').read_text())
    preserved = []
    for original, expected in manifest['hashes'].items():
        p = original.replace('\\', '/')
        if p not in indexed:
            continue
        actual = pointer_hashes.get(p) or hashlib.sha256(git('show', ':'+p)).hexdigest()
        checks['frozen_hash_'+p] = actual == expected
        preserved.append(p)
    stages = git('ls-files', '--stage').decode().splitlines()
    objects = [line.split()[1] for line in stages]
    sizes = git('cat-file', '--batch-check=%(objectsize)', input=('\n'.join(objects)+'\n').encode())
    checks['ordinary_git_blobs_below_100_mib'] = max(map(int, sizes.splitlines())) < 100*1024**2
    result = dict(all_passed=all(checks.values()), checks=checks, tracked_files=len(files),
                  python_files=len(py_files), json_files=len(json_files), frozen_hashes_checked=len(preserved),
                  ordinary_git_bytes=sum(map(int, sizes.splitlines())), broken_links=broken)
    (ROOT/'outputs').mkdir(exist_ok=True)
    (ROOT/'outputs/repository_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='checks'},indent=2))
    if not result['all_passed']:
        raise AssertionError([k for k,v in checks.items() if not v])


if __name__ == '__main__':
    main()
