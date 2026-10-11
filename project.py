"""Repository maintenance entry point; no market requests or trading actions."""
from pathlib import Path
import argparse
import importlib.util
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SUITES = {
    'engineering': ROOT / 'tests',
    'rotation': ROOT / 'outputs/asset_rotation/tests',
    'single': ROOT / 'outputs/single_asset/test_model.py',
}
DEPENDENCIES = {
    'engineering': ['numpy', 'pandas', 'scipy', 'cvxpy', 'sklearn', 'pypfopt', 'pytest'],
    'rotation': ['numpy', 'pandas', 'scipy', 'cvxpy', 'sklearn', 'pypfopt', 'pytest'],
    'single': ['numpy', 'pandas', 'sklearn', 'pytest'],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['inventory', 'doctor', 'test'])
    parser.add_argument('--suite', choices=['all', *SUITES], default='all')
    args = parser.parse_args()
    selected = list(SUITES) if args.suite == 'all' else [args.suite]
    if args.command == 'inventory':
        excluded = {'.git', '.venv', '__pycache__', '.pytest_cache', 'references', 'references_v2',
                    'runs', 'state', 'build', 'dist', 'wheel_verification'}
        files = sorted(p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*.py')
                       if not excluded.intersection(p.relative_to(ROOT).parts))
        print(json.dumps({'root': str(ROOT), 'python_files_excluding_vendor_and_env': files,
                          'test_suites': {k: str(v.relative_to(ROOT)) for k, v in SUITES.items()}},
                         ensure_ascii=False, indent=2))
        return 0
    if args.command == 'doctor':
        modules = sorted({m for s in selected for m in DEPENDENCIES[s]})
        missing = [m for m in modules if importlib.util.find_spec(m) is None]
        print(json.dumps({'python': sys.executable, 'version': sys.version.split()[0],
                          'suites': selected, 'missing_modules': missing,
                          'note': 'Only module presence checked; run test for compatibility. Wind connectivity not checked.'},
                         ensure_ascii=False, indent=2))
        return int(bool(missing))
    # Separate processes keep legacy bare imports from leaking between suites.
    failed = False
    for suite in selected:
        result = subprocess.run([sys.executable, '-m', 'pytest', str(SUITES[suite]), '-q'], cwd=ROOT)
        failed = failed or result.returncode != 0
    return int(failed)


if __name__ == '__main__':
    raise SystemExit(main())
