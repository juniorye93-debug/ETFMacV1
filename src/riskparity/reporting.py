"""Reports are generated solely from one immutable run directory."""
from pathlib import Path
import csv
import html
import json
import math


def report(out):
    out = Path(out)
    metrics = json.loads((out / 'metrics.json').read_text())
    assessment = json.loads((out / 'assessment.json').read_text())
    manifest = json.loads((out / 'run.json').read_text())
    rows = ''.join('<tr><td>' + html.escape(str(k)) + '</td><td>' +
                   html.escape(json.dumps(v, ensure_ascii=False)) + '</td></tr>' for k, v in metrics.items())
    chart = ''
    curve = out / ('curve.csv' if (out / 'curve.csv').exists() else 'equity.csv')
    with curve.open() as stream:
        raw = list(csv.DictReader(stream))
    values = [float(r.get('nav', r.get('equity'))) for r in raw]
    if values and all(math.isfinite(v) for v in values):
        # Sample for display only; metrics always use the complete daily series.
        step = max(1, len(values) // 1500)
        sample = values[::step]
        if len(values) > 1 and (len(values) - 1) % step:
            sample.append(values[-1])
        low, high = min(sample), max(sample)
        points = ' '.join(f'{20+960*i/max(1,len(sample)-1):.2f},{240-210*(v-low)/max(1e-12,high-low):.2f}'
                          for i, v in enumerate(sample))
        chart = f'<svg viewBox="0 0 1000 270" role="img" aria-label="Net asset value"><polyline points="{points}" fill="none" stroke="#2563eb" stroke-width="2"/></svg>'
    cfg = manifest['config']
    title = cfg.get('strategy', cfg.get('symbol', 'Research'))
    content = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>{html.escape(title)} 运行报告</title>
<style>body{{font:16px system-ui;max-width:1050px;margin:40px auto;padding:0 24px;color:#172033}}td{{padding:8px;border-bottom:1px solid #ddd}}svg{{width:100%;background:#f8fafc}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}a{{color:#2563eb}}</style>
<h1>{html.escape(title)} 运行报告</h1><p>数据截止：{html.escape(manifest.get('data_end',''))}；运行：{html.escape(manifest['run_id'])}</p>
<p>状态：<strong>{html.escape(assessment['status'])}</strong>；仅研究，未自动下单。</p>
<p>拒绝/待验证原因：{html.escape(', '.join(assessment.get('reasons', [])) or '历史目标通过，仍需独立纸面交易验证')}</p>
<h2>净值</h2>{chart}<p>横轴为完整评价区间的观测顺序，纵轴范围 {min(values):.6g}—{max(values):.6g}。图仅作展示，详细日期与净值见 CSV。</p>
<h2>指标</h2><table>{rows}</table><h2>运行文件</h2><ul>'''
    for path in sorted(out.iterdir()):
        if path.is_file() and path.name != 'report.html':
            content += f'<li><a href="{html.escape(path.name)}">{html.escape(path.name)}</a></li>'
    content += '</ul><h2>冻结配置</h2><pre>' + html.escape(json.dumps(cfg, ensure_ascii=False, indent=2)) + '</pre></html>'
    (out / 'report.html').write_text(content, encoding='utf-8')


def comparison(root):
    root = Path(root)
    rows, candidates = [], []
    for item in json.loads((root / 'batch.json').read_text())['jobs']:
        p = root / item['name']
        row = {'name': item['name'], 'execution_status': item['status']}
        if (p / 'metrics.json').exists() and item['status'] == 'COMPLETED':
            metrics = json.loads((p / 'metrics.json').read_text())
            assessment = json.loads((p / 'assessment.json').read_text())
            row.update({k: metrics.get(k) for k in ('annual_return', 'annual_excess', 'max_drawdown')})
            row['assessment'] = assessment['status']
            if assessment.get('goal_passed'):
                candidates.append(item['name'])
        rows.append(row)
    from .common import write_json
    write_json(root / 'selection.json', {'historical_goal_candidates': candidates, 'deployment_status': 'NOT_APPROVED',
                                        'rule': 'No automatic winner selection or live promotion; empty means no qualifying candidate'})
    with (root / 'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['name', 'execution_status', 'assessment', 'annual_return', 'annual_excess', 'max_drawdown'])
        writer.writeheader()
        writer.writerows(rows)
    links = ''.join(f'<li>{html.escape(r["name"])}: {html.escape(r["execution_status"])} '
                    f'<a href="{html.escape(r["name"])}/report.html">报告</a></li>' for r in rows)
    (root / 'index.html').write_text('<!doctype html><meta charset="utf-8"><h1>批量研究结果</h1><ul>' + links +
                                    '</ul><p>历史达标候选：' + html.escape(', '.join(candidates) or '无') +
                                    '</p><a href="comparison.csv">对比表</a>', encoding='utf-8')
