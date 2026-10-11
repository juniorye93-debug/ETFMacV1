from pathlib import Path
import datetime as dt
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile


class PipelineError(Exception):
    def __init__(self, code, message):
        super().__init__(message)
        self.code = code


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, 'item'):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (dt.date, Path)):
        return str(value)
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, delete=False) as stream:
        tmp = Path(stream.name)
        json.dump(clean(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def engine(which='rotation'):
    # Legacy bare imports remain internal. Each workflow executes in its own process.
    import riskparity_legacy
    import riskparity_single
    directory = Path((riskparity_legacy if which == 'rotation' else riskparity_single).__file__).parent
    sys.path.insert(0, str(directory))
    return directory


def resolve(config_file, value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (Path(config_file).parent / p).resolve()


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding='utf-8'))
    if cfg.get('schema_version') != 1:
        raise PipelineError('CONFIG_INVALID', 'schema_version must be 1')
    if cfg.get('kind') not in {'portfolio', 'single'}:
        raise PipelineError('CONFIG_INVALID', 'kind must be portfolio or single')
    if cfg.get('market') not in {'CN', 'US'}:
        raise PipelineError('CONFIG_INVALID', 'market must be CN or US')
    if not isinstance(cfg.get('prices'), str):
        raise PipelineError('CONFIG_INVALID', 'prices is required')
    cfg['prices'] = str(resolve(path, cfg['prices']))
    for key in ('cash', 'universe'):
        if key in cfg:
            cfg[key] = str(resolve(path, cfg[key]))
    if cfg['kind'] == 'portfolio':
        if cfg.get('strategy') not in {'erc', 'flat', 'nested_target', 'protective_nested', 'adaptive'}:
            raise PipelineError('CONFIG_INVALID', 'unknown portfolio strategy')
        if not 0 < cfg.get('max_drawdown', 0) < 1:
            raise PipelineError('CONFIG_INVALID', 'max_drawdown must be explicitly set in (0,1)')
        if not all(k in cfg for k in ('cash', 'universe')):
            raise PipelineError('CONFIG_INVALID', 'cash and universe are required')
    elif not cfg.get('symbol'):
        raise PipelineError('CONFIG_INVALID', 'symbol is required')
    if cfg.get('excess_return_multiple', 2) <= 0:
        raise PipelineError('CONFIG_INVALID', 'excess_return_multiple must be positive')
    return cfg


def subprocess_env():
    env = dict(os.environ)
    # These values only bound numerical-library threads, not the statistical model.
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env.setdefault(key, '1')
    env['MPLBACKEND'] = 'Agg'
    return env


def run_process(args, log_path, timeout):
    with Path(log_path).open('w', encoding='utf-8') as log:
        try:
            return subprocess.run([sys.executable, '-m', 'riskparity', *args], stdout=log,
                                  stderr=subprocess.STDOUT, env=subprocess_env(), timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            raise PipelineError('TIMEOUT', f'Worker exceeded {timeout} seconds; see {log_path}')
