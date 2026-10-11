"""Workflow failure and audit invariants; test fixtures never imply investment performance."""
from pathlib import Path
import json
import sqlite3

import pytest

from riskparity.common import PipelineError, load_config, write_json
from riskparity.workflows import evaluate_goals, erc_weights
from riskparity.ledger import replay
from riskparity import cli


def job(tmp_path, **changes):
    config = {'schema_version': 1, 'kind': 'portfolio', 'market': 'CN', 'strategy': 'erc',
              'prices': 'prices.csv', 'cash': 'cash.csv', 'universe': 'universe.json', 'max_drawdown': .08}
    config.update(changes)
    path = tmp_path / 'config.json'
    write_json(path, config)
    return path


def test_paths_are_relative_to_config_not_current_directory(tmp_path, monkeypatch):
    path = job(tmp_path)
    monkeypatch.chdir(tmp_path.parent)
    assert load_config(path)['prices'] == str(tmp_path / 'prices.csv')


@pytest.mark.parametrize('change', [{'strategy': 'typo'}, {'max_drawdown': 0}, {'max_drawdown': 8}, {'schema_version': 2}])
def test_invalid_config_rejected_before_execution(tmp_path, change):
    with pytest.raises(PipelineError, match='unknown|must'):
        load_config(job(tmp_path, **change))


@pytest.mark.parametrize('metrics,reasons', [
    ({'annual_excess': .10, 'max_drawdown': .04}, ['EXCESS_RETURN_SHORTFALL']),
    ({'annual_excess': .20, 'max_drawdown': .10}, ['DRAWDOWN_EXCEEDED']),
    ({'annual_excess': float('nan'), 'max_drawdown': .01}, ['METRICS_INVALID']),
])
def test_no_least_bad_candidate_or_nan_promotion(metrics, reasons):
    result = evaluate_goals(metrics, {'max_drawdown': .08})
    assert result['reasons'] == reasons
    assert not result['goal_passed']
    assert result['deployment_status'] == 'NOT_VALIDATED'


def test_historical_pass_does_not_approve_live_trading():
    result = evaluate_goals({'annual_excess': .16, 'max_drawdown': .08}, {'max_drawdown': .08})
    assert result['goal_passed']
    assert result['deployment_status'] == 'REQUIRES_PAPER_TRADING'


def test_erc_equalizes_risk_instead_of_capital():
    import numpy as np
    w, rc = erc_weights(np.diag([.04, .0004, .01]))
    np.testing.assert_allclose(rc, [1/3]*3, atol=1e-9)
    assert w[1] > w[2] > w[0]
    assert w.sum() == pytest.approx(1)


def run_fixture(tmp_path):
    out = tmp_path / 'run'
    out.mkdir()
    write_json(out / 'run.json', {'status': 'COMPLETED', 'code': {'sha256': 'code-v1'},
                                 'inputs': {'universe': {'sha256': 'universe-v1'}},
                                 'config': {'kind': 'portfolio', 'strategy': 'erc', 'market': 'CN', 'max_drawdown': .08}})
    (out / 'curve.csv').write_text('date,nav\n2020-01-01,1.0\n2020-01-02,1.1\n')
    return out


def test_replay_idempotent_and_appends_only_new_rows(tmp_path):
    out = run_fixture(tmp_path)
    db = tmp_path / 'ledger.sqlite'
    assert replay(out, db, 'test')['added'] == 2
    assert replay(out, db, 'test')['added'] == 0
    with (out / 'curve.csv').open('a') as stream:
        stream.write('2020-01-03,1.2\n')
    assert replay(out, db, 'test')['added'] == 1


def test_revised_history_rolls_back_entire_ledger_transaction(tmp_path):
    out = run_fixture(tmp_path)
    db = tmp_path / 'ledger.sqlite'
    replay(out, db, 'test')
    (out / 'curve.csv').write_text('date,nav\n2020-01-01,1.0\n2020-01-02,0.9\n2020-01-03,1.2\n')
    with pytest.raises(PipelineError) as error:
        replay(out, db, 'test')
    assert error.value.code == 'LEDGER_HISTORY_CHANGED'
    with sqlite3.connect(db) as con:
        assert con.execute('SELECT COUNT(*) FROM observations').fetchone()[0] == 2


def test_changed_model_cannot_mutate_existing_account(tmp_path):
    out = run_fixture(tmp_path)
    db = tmp_path / 'ledger.sqlite'
    replay(out, db, 'test')
    manifest = json.loads((out / 'run.json').read_text())
    manifest['code']['sha256'] = 'code-v2'
    write_json(out / 'run.json', manifest)
    with pytest.raises(PipelineError) as error:
        replay(out, db, 'test')
    assert error.value.code == 'LEDGER_CONFIG_CHANGED'


def test_fetch_starts_wind_and_discards_partial_source_before_fallback(tmp_path, monkeypatch):
    config = job(tmp_path, market='US')
    calls = []
    def worker(args, log, timeout):
        part = args[args.index('--part') + 1]
        provider = args[args.index('--provider') + 1]
        target = Path(args[args.index('--out') + 1])
        calls.append((part, provider))
        if provider == 'wind':
            # Wind returned a partial artifact and failed. It must not reach final snapshot.
            target.write_text('PARTIAL WIND DATA')
            log.write_text('Wind timeout')
            return 1
        target.write_text('COMPLETE FALLBACK SERIES')
        write_json(target.with_suffix('.meta.json'), {'source': provider, 'synthetic': False})
        return 0
    monkeypatch.setattr(cli, 'run_process', worker)
    dest = tmp_path / 'snapshot'
    cli.fetch_config(config, dest, '2020-01-01', '2021-01-01', 1)
    assert calls == [('prices', 'wind'), ('prices', 'yahoo'), ('cash', 'wind'), ('cash', 'fred')]
    assert (dest / 'prices.csv').read_text() == 'COMPLETE FALLBACK SERIES'
    receipt = json.loads((dest / 'prices.meta.json').read_text())
    assert receipt['fallback_used']
    assert receipt['attempts'][0]['reason']
    assert json.loads((dest / 'cash_request.json').read_text())['start'] == '2019-12-01'


def test_all_fetch_failures_do_not_reuse_old_snapshot(tmp_path, monkeypatch):
    config = job(tmp_path)
    old = tmp_path / 'prices.csv'
    old.write_text('UNCHANGED OLD DATA')
    def failure(args, log, timeout):
        log.write_text('No verified observations')
        return 1
    monkeypatch.setattr(cli, 'run_process', failure)
    dest = tmp_path / 'snapshot'
    with pytest.raises(PipelineError) as error:
        cli.fetch_config(config, dest, '2020-01-01', '2021-01-01', 1)
    assert error.value.code == 'FETCH_FAILED'
    assert old.read_text() == 'UNCHANGED OLD DATA'
    assert not (dest / 'job.json').exists()
    assert json.loads((dest / 'snapshot.json').read_text())['status'] == 'FAILED'


def test_synthetic_data_cannot_enter_formal_pipeline():
    from riskparity.data import check_real
    with pytest.raises(PipelineError):
        check_real({'source': 'software fixture', 'synthetic': True})


def test_stale_price_rejected_before_signal_generation():
    import pandas as pd
    from riskparity.data import check_freshness
    with pytest.raises(PipelineError) as error:
        check_freshness(pd.Timestamp('2020-01-01'), {'as_of': '2020-01-10', 'max_price_age_days': 3})
    assert error.value.code == 'PRICE_STALE'


def test_existing_output_cannot_be_overwritten(tmp_path):
    dest = tmp_path / 'previous-run'
    dest.mkdir()
    sentinel = dest / 'curve.csv'
    sentinel.write_text('OLD RESULTS')
    with pytest.raises(PipelineError) as error:
        cli.make_run(job(tmp_path), dest)
    assert error.value.code == 'OUTPUT_EXISTS'
    assert sentinel.read_text() == 'OLD RESULTS'


def test_daily_uses_completed_day_cutoff_and_is_idempotent(tmp_path, monkeypatch):
    config = job(tmp_path, start='2015-01-01')
    calls = []
    def fake_fetch(config_path, dest, start, end, timeout):
        calls.append((start, end))
        return config_path
    def fake_run(config_path, dest):
        write_json(Path(dest) / 'run.json', {'status': 'COMPLETED'})
    monkeypatch.setattr(cli, 'fetch_config', fake_fetch)
    monkeypatch.setattr(cli, 'make_run', fake_run)
    root = tmp_path / 'daily'
    cli.daily(config, root, '2026-10-07', 15)
    cli.daily(config, root, '2026-10-07', 15)
    assert calls == [('2015-01-01', '2026-10-06')]
    job(tmp_path, start='2015-01-01', max_drawdown=.10)
    with pytest.raises(PipelineError) as error:
        cli.daily(config, root, '2026-10-07', 15)
    assert error.value.code == 'DAILY_CONFIG_CHANGED'


def test_worker_failure_writes_terminal_state_and_preserves_input(tmp_path, monkeypatch):
    config = job(tmp_path)
    for name in ('prices.csv', 'cash.csv', 'universe.json'):
        (tmp_path / name).write_text('TEST-ONLY INPUT')
        if name.endswith('.csv'):
            write_json((tmp_path / name).with_suffix('.meta.json'), {'synthetic': True})
    def fail(args, log, timeout):
        log.write_text('Test injected worker failure')
        write_json(Path(args[args.index('--out') + 1]) / 'error.json', {'code': 'DATA_INVALID', 'message': 'injected'})
        return 2
    monkeypatch.setattr(cli, 'run_process', fail)
    out = tmp_path / 'failed-run'
    with pytest.raises(PipelineError):
        cli.make_run(config, out)
    receipt = json.loads((out / 'run.json').read_text())
    assert receipt['status'] == 'FAILED'
    assert receipt['error']['code'] == 'DATA_INVALID'
    assert (tmp_path / 'prices.csv').read_text() == 'TEST-ONLY INPUT'
    assert (out / 'inputs/prices.csv').read_text() == 'TEST-ONLY INPUT'


def test_input_mutation_cannot_produce_completed_run(tmp_path, monkeypatch):
    config = job(tmp_path)
    for name in ('prices.csv', 'cash.csv', 'universe.json'):
        (tmp_path / name).write_text('TEST-ONLY INPUT')
        if name.endswith('.csv'):
            write_json((tmp_path / name).with_suffix('.meta.json'), {'synthetic': True})
    def tamper(args, log, timeout):
        out = Path(args[args.index('--out') + 1])
        (out / 'inputs/prices.csv').write_text('MUTATED')
        write_json(out / 'result.json', {})
        return 0
    monkeypatch.setattr(cli, 'run_process', tamper)
    out = tmp_path / 'tampered-run'
    with pytest.raises(PipelineError) as error:
        cli.make_run(config, out)
    assert error.value.code == 'INPUT_CHANGED'
    assert json.loads((out / 'run.json').read_text())['status'] == 'FAILED'


def test_source_revision_during_run_is_not_promoted(tmp_path, monkeypatch):
    from riskparity import workflows
    config = job(tmp_path)
    for name in ('prices.csv', 'cash.csv', 'universe.json'):
        (tmp_path / name).write_text('TEST-ONLY INPUT')
        if name.endswith('.csv'):
            write_json((tmp_path / name).with_suffix('.meta.json'), {'synthetic': True})
    versions = iter(['initial-code', 'changed-code'])
    monkeypatch.setattr(workflows, 'code_receipt', lambda snapshot=None: {'sha256': next(versions)})
    def worker(args, log, timeout):
        write_json(Path(args[args.index('--out') + 1]) / 'result.json', {})
        return 0
    monkeypatch.setattr(cli, 'run_process', worker)
    with pytest.raises(PipelineError) as error:
        cli.make_run(config, tmp_path / 'run')
    assert error.value.code == 'CODE_CHANGED'
