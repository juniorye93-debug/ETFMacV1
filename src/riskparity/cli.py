from pathlib import Path
import argparse
import datetime as dt
import importlib.metadata
import importlib.util
import json
import shutil
import sys
import tempfile
import uuid

from .common import PipelineError, digest, load_config, now, resolve, run_process, write_json


def make_run(config, destination):
    cfg = load_config(config)
    out = Path(destination).resolve()
    try:
        out.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise PipelineError('OUTPUT_EXISTS', f'Use a new output directory: {out}')
    from .workflows import code_receipt
    manifest = {'schema_version': 1, 'run_id': uuid.uuid4().hex, 'status': 'RUNNING', 'started_at_utc': now(),
                'config': cfg, 'config_file': str(Path(config).resolve()), 'python': sys.version,
                'inputs': {}, 'code': code_receipt(out / 'code'), 'research_only': True,
                'dependencies': {d.metadata['Name']: d.version for d in importlib.metadata.distributions()}}
    write_json(out / 'job.json', cfg)
    write_json(out / 'run.json', manifest)
    try:
        for key in ('prices', 'cash', 'universe'):
            if key in cfg:
                p = Path(cfg[key])
                expected = digest(p)
                frozen = out / 'inputs' / (key + p.suffix)
                frozen.parent.mkdir(exist_ok=True)
                shutil.copyfile(p, frozen)
                if digest(frozen) != expected:
                    raise PipelineError('INPUT_CHANGED', f'Input changed while snapshotting: {p}')
                manifest['inputs'][key] = {'path': str(frozen), 'original_path': str(p), 'sha256': expected}
                if key != 'universe':
                    meta = p.with_suffix('.meta.json')
                    manifest['inputs'][key]['metadata_sha256'] = digest(meta)
                    shutil.copyfile(meta, frozen.with_suffix('.meta.json'))
                    if digest(frozen.with_suffix('.meta.json')) != manifest['inputs'][key]['metadata_sha256']:
                        raise PipelineError('INPUT_CHANGED', 'Provenance changed while snapshotting')
                cfg[key] = str(frozen)
        write_json(out / 'job.json', cfg)
        timeout = cfg.get('timeout_seconds', 7200)
        if timeout <= 0:
            raise PipelineError('CONFIG_INVALID', 'timeout_seconds must be positive')
        write_json(out / 'run.json', manifest)
        code = run_process(['_run', '--config', str(out / 'job.json'), '--out', str(out)], out / 'run.log', timeout)
        if code:
            error = json.loads((out / 'error.json').read_text()) if (out / 'error.json').exists() else {
                'code': 'WORKER_FAILED', 'message': f'Worker exited {code}; see run.log'}
            raise PipelineError(error['code'], error['message'])
        manifest.update(json.loads((out / 'result.json').read_text()))
        # Inputs must remain immutable throughout the run.
        for receipt in manifest['inputs'].values():
            if digest(receipt['path']) != receipt['sha256']:
                raise PipelineError('INPUT_CHANGED', f'Input changed during run: {receipt["path"]}')
            if receipt.get('metadata_sha256') and digest(Path(receipt['path']).with_suffix('.meta.json')) != receipt['metadata_sha256']:
                raise PipelineError('INPUT_CHANGED', 'Input provenance changed during run')
        if code_receipt()['sha256'] != manifest['code']['sha256']:
            raise PipelineError('CODE_CHANGED', 'Runtime source changed during run; results are not promoted')
        manifest.update(status='COMPLETED', finished_at_utc=now())
        write_json(out / 'run.json', manifest)
        from .reporting import report
        report(out)
        print(json.dumps({'run': str(out), 'status': 'COMPLETED',
                          'assessment': json.loads((out / 'assessment.json').read_text()),
                          'report': str(out / 'report.html')}, ensure_ascii=False, indent=2))
        return out
    except BaseException as exc:
        error = {'code': 'INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else getattr(exc, 'code', 'RUN_FAILED'), 'message': str(exc)}
        manifest.update(status='INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'FAILED', finished_at_utc=now(), error=error)
        write_json(out / 'error.json', error)
        write_json(out / 'run.json', manifest)
        raise


def fetch_config(config, destination, start, end, timeout):
    cfg = load_config(config)
    if dt.date.fromisoformat(start) >= dt.date.fromisoformat(end) or timeout <= 0:
        raise PipelineError('CONFIG_INVALID', 'Invalid fetch interval/timeout')
    out = Path(destination).resolve()
    if out.exists():
        raise PipelineError('OUTPUT_EXISTS', 'Fetch requires a new snapshot directory')
    out.mkdir(parents=True)
    request = {**cfg, 'start': start, 'end': end}
    write_json(out / 'request.json', request)
    parts = ['single'] if cfg['kind'] == 'single' else ['prices', 'cash']
    receipts = {}
    try:
        for part in parts:
            part_request = dict(request)
            if part == 'cash':
                # Cash availability lags require observations before the first price date.
                part_request['start'] = (dt.date.fromisoformat(start) - dt.timedelta(days=31)).isoformat()
            request_file = out / f'{part}_request.json'
            write_json(request_file, part_request)
            providers = ['wind']
            if part == 'cash':
                providers += ['fred'] if cfg['market'] == 'US' else ['exact_series_fallback']
            else:
                providers += ['akshare', 'yahoo'] if cfg['market'] == 'CN' else ['yahoo']
            attempts = []
            success = False
            for provider in providers:
                with tempfile.TemporaryDirectory(prefix='candidate-', dir=out) as temp:
                    candidate = Path(temp) / ('prices.csv' if part != 'cash' else 'cash.csv')
                    log = out / f'{part}_{provider}.log'
                    print(f'Fetching {part}: {provider}', flush=True)
                    try:
                        code = run_process(['_fetch', '--request', str(request_file), '--part', part,
                                            '--provider', provider, '--out', str(candidate)], log, timeout)
                        if code or not candidate.exists() or not candidate.with_suffix('.meta.json').exists():
                            raise PipelineError('PROVIDER_FAILED', log.read_text()[-2000:])
                        receipt = json.loads(candidate.with_suffix('.meta.json').read_text())
                        attempts.append({'provider': provider, 'success': True, 'retrieved_at_utc': now()})
                        receipt.update(attempts=attempts, fallback_used=provider != 'wind')
                        target = out / candidate.name
                        candidate.replace(target)
                        write_json(target.with_suffix('.meta.json'), receipt)
                        receipts[part] = receipt
                        success = True
                        break
                    except Exception as exc:
                        attempts.append({'provider': provider, 'success': False, 'reason': str(exc), 'attempted_at_utc': now()})
                        write_json(out / f'{part}_attempts.json', attempts)
            write_json(out / f'{part}_attempts.json', attempts)
            if not success:
                raise PipelineError('FETCH_FAILED', f'All verified {part} providers failed; see {out}')
        job = dict(cfg)
        job['prices'] = str(out / 'prices.csv')
        if cfg['kind'] == 'portfolio':
            job['cash'] = str(out / 'cash.csv')
        # Fetch interval is also the research interval; no implicit use of old data.
        job.update(start=start, end=end)
        write_json(out / 'job.json', job)
        write_json(out / 'snapshot.json', {'status': 'COMPLETED', 'receipts': receipts, 'finished_at_utc': now()})
        print(f'Snapshot ready. Run: riskparity run --config "{out / "job.json"}"')
        return out / 'job.json'
    except Exception as exc:
        write_json(out / 'snapshot.json', {'status': 'FAILED', 'code': getattr(exc, 'code', 'FETCH_FAILED'),
                                         'message': str(exc), 'finished_at_utc': now()})
        raise


def batch(config, destination):
    config = Path(config).resolve()
    specification = json.loads(config.read_text())
    if specification.get('schema_version') != 1 or not specification.get('jobs'):
        raise PipelineError('CONFIG_INVALID', 'Batch requires schema_version=1 and nonempty jobs')
    names = [j['name'] for j in specification['jobs']]
    import re
    if len(names) != len(set(names)) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', n) for n in names):
        raise PipelineError('CONFIG_INVALID', 'Unique safe job names required')
    jobs = [(j['name'], resolve(config, j['config'])) for j in specification['jobs']]
    for _, path in jobs:
        load_config(path)
    out = Path(destination).resolve()
    if out.exists():
        raise PipelineError('OUTPUT_EXISTS', 'Batch requires a new output directory')
    out.mkdir(parents=True)
    status = {'status': 'RUNNING', 'started_at_utc': now(), 'jobs': []}
    failed = False
    for name, path in jobs:
        print(f'Running {name}', flush=True)
        try:
            make_run(path, out / name)
            status['jobs'].append({'name': name, 'status': 'COMPLETED'})
        except Exception as exc:
            failed = True
            status['jobs'].append({'name': name, 'status': 'FAILED', 'reason': str(exc)})
        write_json(out / 'batch.json', status)
    status.update(status='PARTIAL_FAILURE' if failed else 'COMPLETED', finished_at_utc=now())
    write_json(out / 'batch.json', status)
    from .reporting import comparison
    comparison(out)
    print(f'Batch report: {out / "index.html"}')
    return int(failed)


def daily(config, root, as_of, max_age):
    """Calendar-driven research refresh. Never request today's incomplete daily close."""
    cfg = load_config(config)
    if not cfg.get('start'):
        raise PipelineError('CONFIG_INVALID', 'daily requires start (including warm-up) in config')
    if max_age < 1:
        raise PipelineError('CONFIG_INVALID', 'max-age-days must be positive')
    today = dt.date.fromisoformat(as_of) if as_of else dt.datetime.now(dt.timezone(dt.timedelta(hours=8))).date()
    end = (today - dt.timedelta(days=1)).isoformat()
    cfg.update(end=end, as_of=today.isoformat(), max_price_age_days=max_age)
    name = Path(config).stem
    out = Path(root).resolve() / f'{today.isoformat()}-{name}'
    if out.exists():
        state = out / 'run' / 'run.json'
        if state.exists() and json.loads(state.read_text()).get('status') == 'COMPLETED':
            if json.loads((out / 'config.json').read_text()) != cfg:
                raise PipelineError('DAILY_CONFIG_CHANGED', 'Daily configuration changed; use a new out-root')
            print(json.dumps({'status': 'ALREADY_COMPLETED', 'run': str(out / 'run')}, indent=2))
            return
        raise PipelineError('DAILY_ALREADY_STARTED', f'Existing daily attempt is incomplete: {out}; review it or use another out-root')
    out.mkdir(parents=True)
    write_json(out / 'config.json', cfg)
    refreshed = fetch_config(out / 'config.json', out / 'data', cfg['start'], end, cfg.get('fetch_timeout_seconds', 120))
    make_run(refreshed, out / 'run')


def main():
    parser = argparse.ArgumentParser(description='独立量化研究程序：取数、回测、验收、报告；不依赖 AI，不自动下单')
    subs = parser.add_subparsers(dest='command', required=True)
    subs.add_parser('doctor', help='Check installed engine dependencies and SDK presence')
    run = subs.add_parser('run', help='Run a frozen research configuration')
    run.add_argument('--config', required=True)
    run.add_argument('--out', default=None)
    run.add_argument('--refresh', action='store_true', help='Fetch a new snapshot before running; requires start/end')
    b = subs.add_parser('batch', help='Run configured jobs and build comparison')
    b.add_argument('--config', required=True)
    b.add_argument('--out', default=None)
    fetch = subs.add_parser('fetch', help='Wind-first validated data snapshot')
    fetch.add_argument('--config', required=True)
    fetch.add_argument('--out', required=True)
    fetch.add_argument('--start', required=True)
    fetch.add_argument('--end', required=True)
    fetch.add_argument('--timeout', type=int, default=120)
    validate = subs.add_parser('validate', help='Validate configuration and offline data without backtesting')
    validate.add_argument('--config', required=True)
    replay = subs.add_parser('replay', help='Idempotently append a completed historical run to a research ledger')
    replay.add_argument('--run', required=True)
    replay.add_argument('--db', default='state/research.sqlite')
    replay.add_argument('--account', required=True)
    daily_parser = subs.add_parser('daily', help='One idempotent research refresh per Shanghai calendar day; uses previous-day cutoff')
    daily_parser.add_argument('--config', required=True)
    daily_parser.add_argument('--out-root', default='runs/daily')
    daily_parser.add_argument('--as-of', default=None)
    daily_parser.add_argument('--max-age-days', type=int, default=15)
    worker = subs.add_parser('_run', help=argparse.SUPPRESS)
    worker.add_argument('--config', required=True)
    worker.add_argument('--out', required=True)
    fw = subs.add_parser('_fetch', help=argparse.SUPPRESS)
    fw.add_argument('--request', required=True)
    fw.add_argument('--part', choices=['single', 'prices', 'cash'], required=True)
    fw.add_argument('--provider', required=True)
    fw.add_argument('--out', required=True)
    args = parser.parse_args()
    try:
        if args.command == 'doctor':
            names = ['numpy', 'pandas', 'scipy', 'cvxpy', 'sklearn', 'pypfopt', 'matplotlib']
            missing = [n for n in names if importlib.util.find_spec(n) is None]
            from .common import engine
            engine()
            print(json.dumps({'python': sys.executable, 'missing': missing, 'WindPy_present': importlib.util.find_spec('WindPy') is not None,
                              'note': 'Offline research does not require Wind. SDK presence does not prove connectivity.'}, indent=2))
            return int(bool(missing))
        if args.command == 'run':
            out = args.out or str(Path('runs') / (dt.datetime.now().strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:8]))
            config = args.config
            if args.refresh:
                cfg = load_config(config)
                if not cfg.get('start') or not cfg.get('end'):
                    raise PipelineError('CONFIG_INVALID', '--refresh requires explicit start/end in config')
                config = fetch_config(config, str(Path(out).resolve()) + '-data', cfg['start'], cfg['end'], cfg.get('fetch_timeout_seconds', 120))
            make_run(config, out)
        elif args.command == 'batch':
            return batch(args.config, args.out or str(Path('runs') / ('batch-' + uuid.uuid4().hex[:12])))
        elif args.command == 'fetch':
            fetch_config(args.config, args.out, args.start, args.end, args.timeout)
        elif args.command == 'validate':
            cfg = load_config(args.config)
            from .data import load_portfolio, metadata, check_real, check_freshness
            if cfg['kind'] == 'portfolio':
                p, *_ = load_portfolio(cfg)
            else:
                from .common import engine
                engine('single')
                import model
                check_real(metadata(cfg['prices']))
                p = model.read_data(cfg['prices'])
                if cfg.get('start'):
                    p = p.loc[cfg['start']:]
                if cfg.get('end'):
                    p = p.loc[:cfg['end']]
                if p.empty:
                    raise PipelineError('DATA_INVALID', 'Empty selected interval')
                check_freshness(p.index[-1], cfg)
            print(json.dumps({'status': 'VALID', 'rows': len(p), 'data_start': str(p.index[0].date()), 'data_end': str(p.index[-1].date())}, indent=2))
        elif args.command == 'replay':
            from .ledger import replay
            print(json.dumps(replay(args.run, args.db, args.account), indent=2))
        elif args.command == 'daily':
            daily(args.config, args.out_root, args.as_of, args.max_age_days)
        elif args.command == '_run':
            cfg = load_config(args.config)
            from .workflows import run_portfolio, run_single
            result = (run_portfolio if cfg['kind'] == 'portfolio' else run_single)(cfg, Path(args.out))
            write_json(Path(args.out) / 'result.json', result)
        elif args.command == '_fetch':
            request = json.loads(Path(args.request).read_text())
            from .data import worker_prices, worker_cash
            if args.part == 'single':
                from .common import engine
                engine('single')
                import fetch
                from types import SimpleNamespace
                import model
                a = SimpleNamespace(provider=args.provider, symbol=request['symbol'], start=request['start'], end=request['end'],
                                    calendar=request.get('calendar', 'SSE' if request['market'] == 'CN' else 'NYSE'),
                                    adjustment='F', instrument=request.get('instrument', 'etf'))
                frame, receipt = fetch.provider_fetch(a)
                frame = frame.loc[a.start:a.end]
                # Let the existing exporter normalize pre-listing missing OHLC and validate.
                a.out = str(Path(args.out).parent / 'single')
                receipt = fetch.persist(a, frame, receipt)
                original = Path(a.out) / f'{a.symbol}.csv'
                model.read_data(original)
                shutil.copyfile(original, args.out)
                write_json(Path(args.out).with_suffix('.meta.json'), receipt)
            else:
                (worker_prices if args.part == 'prices' else worker_cash)(request, args.provider, args.out)
        return 0
    except KeyboardInterrupt:
        print('Interrupted. Existing research results were preserved.', file=sys.stderr)
        return 130
    except Exception as exc:
        error = {'code': getattr(exc, 'code', 'PROGRAM_ERROR'), 'message': str(exc)}
        if args.command == '_run':
            write_json(Path(args.out) / 'error.json', error)
        print(json.dumps(error, ensure_ascii=False), file=sys.stderr)
        return 2
