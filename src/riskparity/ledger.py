"""Idempotent research replay ledger; this is not a broker or live order executor."""
from pathlib import Path
import csv
import hashlib
import json
import sqlite3

from .common import PipelineError


def replay(run, database, account):
    run = Path(run)
    manifest = json.loads((run / 'run.json').read_text())
    if manifest['status'] != 'COMPLETED':
        raise PipelineError('RUN_INCOMPLETE', 'Only completed runs can be replayed')
    cfg = manifest['config']
    fingerprint_config = {k: v for k, v in cfg.items() if k not in {'prices', 'cash', 'end', 'as_of', 'max_price_age_days', 'max_cash_age_days'}}
    fingerprint_config['universe'] = manifest['inputs'].get('universe', {}).get('sha256')
    payload = {'config': fingerprint_config, 'code': manifest['code']['sha256']}
    fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    curve_file = run / ('curve.csv' if (run / 'curve.csv').exists() else 'equity.csv')
    with curve_file.open() as stream:
        curve = list(csv.DictReader(stream))
    rows = [(r.get('date', next(iter(r.values()))), json.dumps(r, sort_keys=True)) for r in curve]
    if not rows or len({d for d, _ in rows}) != len(rows):
        raise PipelineError('LEDGER_INVALID', 'Curve must have unique daily observations')
    Path(database).parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(database, timeout=30) as db:
        db.execute('CREATE TABLE IF NOT EXISTS accounts(account TEXT PRIMARY KEY,fingerprint TEXT NOT NULL)')
        db.execute('CREATE TABLE IF NOT EXISTS observations(account TEXT,date TEXT,payload TEXT NOT NULL,PRIMARY KEY(account,date))')
        # One transaction for identity checks, all overlap checks and inserts.
        db.execute('BEGIN IMMEDIATE')
        previous = db.execute('SELECT fingerprint FROM accounts WHERE account=?', (account,)).fetchone()
        if previous and previous[0] != fingerprint:
            raise PipelineError('LEDGER_CONFIG_CHANGED', 'Use a new account name for changed strategy/config/code')
        db.execute('INSERT OR IGNORE INTO accounts VALUES(?,?)', (account, fingerprint))
        existing = dict(db.execute('SELECT date,payload FROM observations WHERE account=?', (account,)))
        if existing and rows[0][0] != min(existing):
            raise PipelineError('LEDGER_HISTORY_CHANGED', 'Replay must begin at the same original evaluation date')
        # Require complete prior history; prevents inserting over missing/revised observations.
        incoming = dict(rows)
        if any(incoming.get(date) != value for date, value in existing.items()):
            raise PipelineError('LEDGER_HISTORY_CHANGED', 'Previously recorded history differs or is missing; no records changed')
        added = 0
        for date, value in rows:
            if date not in existing:
                db.execute('INSERT INTO observations VALUES(?,?,?)', (account, date, value))
                added += 1
        total = db.execute('SELECT COUNT(*) FROM observations WHERE account=?', (account,)).fetchone()[0]
    return {'account': account, 'added': added, 'total': total, 'last_date': rows[-1][0],
            'mode': 'historical_research_replay', 'orders_sent': 0}
