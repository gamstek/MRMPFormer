"""Generate a self-contained HTML report from saved three-way inference results."""
import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path


def pretty(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(
        '<th>' + escape(str(h)) + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + escape(str(v)) + '</td>' for v in row) + '</tr>' for row in rows
    ) + '</tbody></table></div>'


def peak_chart(item, result, label, color):
    """Draw returned RT intervals on the unchanged input signal."""
    x, y = item['x'], item['y']
    xmin, xmax, ymin, ymax = min(x), max(x), min(y), max(y)
    def sx(value):
        return 60 + (value - xmin) / (xmax - xmin or 1) * 860
    def sy(value):
        return 200 - (value - ymin) / (ymax - ymin or 1) * 150
    points = ' '.join(f'{sx(a):.3f},{sy(b):.3f}' for a, b in zip(x, y))
    parts = [f'<svg class="peak-chart" viewBox="0 0 980 250" role="img" aria-label="{escape(label)} 峰区间">',
             f'<text x="60" y="20">{escape(label)} · {len(result["peaks"])} 个峰</text>',
             '<path d="M60 45V200H925" stroke="#a9b7c5" fill="none"/>']
    for index, peak in enumerate(result['peaks']):
        a, b, c = peak['a'], peak['b'], peak['c']
        left, right = sx(max(xmin, min(xmax, a))), sx(max(xmin, min(xmax, b)))
        # Offset annotation rows so overlapping intervals remain identifiable.
        top = 42 + (index % 3) * 16
        title = escape(f'峰 {index + 1}: a={a}, b={b}, c={c}')
        parts.append(f'<g class="peak-box"><title>{title}</title>'
                     f'<rect x="{left:.3f}" y="{top}" width="{max(right-left, .5):.3f}" '
                     f'height="{200-top}" fill="{color}" fill-opacity="0.09" stroke="{color}" stroke-width="1.5"/>'
                     f'<text x="{min(left + 3, 800):.3f}" y="{top-4}" style="fill:{color}">'
                     f'#{index+1} c={c:.4g}</text></g>')
    parts.append(f'<polyline points="{points}" fill="none" stroke="#536779" stroke-width="1.3"/>')
    for i in range(6):
        rt = xmin + (xmax-xmin)*i/5
        parts.append(f'<text x="{sx(rt):.3f}" y="223" text-anchor="middle">{rt:.4g}</text>')
    parts.append('<text x="920" y="243" text-anchor="end">保留时间 / min</text>')
    if not result['peaks']:
        parts.append('<text x="700" y="40">未返回峰框</text>')
    parts.append('</svg>')
    return ''.join(parts)


def render(output):
    output = Path(output)
    report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
    cases = []
    for record in report['records']:
        path = output / (record['uid'] + '.json')
        if path.parent.resolve() != output.resolve():
            raise ValueError('Invalid case identifier')
        case = json.loads(path.read_text(encoding='utf-8'))
        for mode in ('batch', 'single'):
            for side in ('python', 'build', 'c'):
                if side + '_' + mode not in case:
                    raise ValueError(f'{path}: missing {side}_{mode}; run three-way tests first')
        if len(case['input']['x']) != len(case['input']['y']):
            raise ValueError(f'{path}: input array lengths differ')
        cases.append(dict(summary=record, **case))
    if len(cases) != report['cases']:
        raise ValueError('Report case count mismatch')
    body = []
    for index, case in enumerate(cases):
        record, item = case['summary'], case['input']
        uid = escape(record['uid'])
        passed = record['passed']
        body.append(f'<details class="case" data-uid="{uid}" data-passed="{str(passed).lower()}"><summary>'
                    f'<span class="badge {"pass" if passed else "fail"}">{"通过" if passed else "失败"}</span> '
                    f'<strong>{uid}</strong><span>{len(item["x"])} 点 · 峰数 {record.get("peaks", "未记录")} · '
                    f'最大差异 {record.get("max_absolute_delta", "未记录")}</span></summary><div class="content">')
        if record.get('error'):
            body.append('<pre class="fail">' + escape(record['error']) + '</pre>')
        body.append(f'<button onclick="downloadCase({index})">下载本例完整 JSON</button><h3>输入参数</h3>')
        body.append(table(['配置项', '值'], case['config'].items()))
        body.append(table(['输入字段', '值'], [(k, v) for k, v in item.items() if k not in ('x', 'y')]))
        x, y = item['x'], item['y']
        xmin, xmax, ymin, ymax = min(x), max(x), min(y), max(y)
        points = ' '.join(f'{40 + (a-xmin)/(xmax-xmin or 1)*900:.3f},{200-(b-ymin)/(ymax-ymin or 1)*170:.3f}' for a, b in zip(x, y))
        body.append(f'<h3>输入曲线 · {len(x)} 个采样点</h3><svg viewBox="0 0 980 240" role="img" aria-label="{uid} 输入强度曲线">'
                    '<path d="M40 25V200H945" stroke="#a9b7c5" fill="none"/>'
                    f'<polyline points="{points}" fill="none" stroke="#167e91" stroke-width="1.5"/>'
                    f'<text x="40" y="225">RT {xmin}</text><text x="840" y="225">{xmax}</text>'
                    f'<text x="40" y="18">强度 {ymin} ~ {ymax}</text></svg>')
        body.append('<details><summary>完整 x / y 数据表（无截断）</summary>')
        body.append(table(['点序号（从 0 开始）', 'x / RT', 'y / 强度'], ((i, a, b) for i, (a, b) in enumerate(zip(x, y)))))
        body.append('</details>')
        for mode, label in [('batch', '批量调用'), ('single', '单条调用')]:
            sides = [case[prefix + '_' + mode] for prefix in ('python', 'build', 'c')]
            body.append(f'<h3>{label} · 三方完整对照</h3>')
            body.append('<p>框的左右边界为 a / b，标注为峰序号和 c 值（简写）。悬停峰框查看完整数值；三图使用相同坐标范围。</p>')
            for side, name, color in zip(sides, ('Python 源码', 'Cython build', 'C DLL'),
                                         ('#167e91', '#a36017', '#7054a0')):
                body.append(peak_chart(item, side, name, color))
            body.append(table(['字段', 'Python 源码', 'Cython build', 'C DLL'],
                              [[key] + [json.dumps(side.get(key), ensure_ascii=False) for side in sides]
                               for key in ('uid', 'status', 'alerts')] + [['峰数'] + [len(side['peaks']) for side in sides]]))
            rows = []
            for i in range(max(len(side['peaks']) for side in sides)):
                for key in ('a', 'b', 'c'):
                    values = [side['peaks'][i].get(key) if i < len(side['peaks']) else None for side in sides]
                    delta = [abs(values[a] - values[b]) if values[a] is not None and values[b] is not None else '缺失'
                             for a, b in ((0, 1), (0, 2), (1, 2))]
                    rows.append([i + 1, key] + ['缺失' if v is None else v for v in values] + delta)
            body.append(table(['峰', '字段', 'Python 源码', 'Cython build', 'C DLL', '|Py−build|', '|Py−C|', '|build−C|'], rows)
                        if rows else '<p>三方均未返回峰；状态和完整告警见上表。</p>')
        body.append('<details><summary>完整原始 JSON（输入、配置、三方六组输出）</summary><pre>' + escape(pretty(case)) + '</pre></details></div></details>')
    payload = dict(generated_at=datetime.now(timezone.utc).isoformat(), report=report, cases=cases)
    data = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MRMPFormer 三方推理对照报告</title><style>
:root{font-family:Segoe UI,Microsoft YaHei,sans-serif;color:#243448;background:#edf2f6;font-size:15px}*{box-sizing:border-box}body{margin:0}main{max-width:1280px;margin:auto;padding:36px 24px}h1{font-size:30px;margin:8px 0 18px}h3{margin-top:28px}.eyebrow{color:#167e91;font-weight:700;letter-spacing:2px}.metrics{display:flex;gap:30px;flex-wrap:wrap;padding:20px 0;border-block:1px solid #c9d3dd}.metrics strong{font-size:28px;display:block}header p{line-height:1.7}.toolbar{display:flex;gap:12px;flex-wrap:wrap;align-items:center;padding:22px 0}input,select,button{font:inherit;border:1px solid #aab8c6;border-radius:5px;background:white;padding:9px 12px;color:inherit}button{cursor:pointer}button:hover{background:#e0f0f3}:focus-visible{outline:3px solid #167e91;outline-offset:3px}.case{background:white;margin:10px 0;border:1px solid #ccd6e0;border-radius:6px}.case>summary{display:flex;gap:14px;align-items:center;padding:16px;flex-wrap:wrap;cursor:pointer}.case>summary:before{content:'▸'}.case[open]>summary:before{content:'▾'}.case>summary>span:last-child{margin-left:auto;color:#536779}.content{padding:0 20px 24px}.badge{font-size:12px;font-weight:700;padding:3px 8px;border-radius:3px}.pass{color:#126341;background:#e4f4eb}.fail{color:#a22b32;background:#fff0ef}.scroll{overflow:auto;max-height:460px;margin:12px 0;border:1px solid #d7dfe7}table{width:100%;border-collapse:collapse;font-size:13px;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:10px 12px;border-bottom:1px solid #e2e8ee;vertical-align:top;white-space:pre-wrap;overflow-wrap:anywhere}th{background:#edf3f8;position:sticky;top:0}tbody tr:nth-child(even){background:#f8fafc}td{font-family:Consolas,monospace}pre{overflow:auto;max-height:550px;padding:16px;background:#f3f6f9;font-size:12px;line-height:1.5}svg{width:100%;height:auto;background:#f7fafc;margin-bottom:16px}svg text{font-size:12px;fill:#536779}details details>summary{padding:12px 0;cursor:pointer}#empty{padding:30px;text-align:center}[hidden]{display:none!important}@media(max-width:600px){main{padding:20px 10px}h1{font-size:24px}.content{padding:0 10px 16px}.case>summary>span:last-child{margin-left:0}}@media print{.toolbar,button{display:none}.scroll,pre{max-height:none;overflow:visible}body{background:white}}
</style><main><header><div class="eyebrow">MRMPFORMER / INFERENCE VALIDATION</div><h1>三方推理 · 完整数据报告</h1>
<p>Python 源码 / model/build 中的 Cython 扩展 / 集成包 C DLL。相同输入和配置，批量与单条分别对照。</p>
<div class="metrics">METRICS</div><p>数据均嵌入本文件，可离线打开或发送给他人。原始数值不做显示舍入；曲线坐标仅作可视化，完整精度见数据表和 JSON。</p></header>
<div class="toolbar"><label>查找用例 <input id="search" type="search" placeholder="例如 test3_1_000"></label><label>结果 <select id="filter"><option value="all">全部</option><option value="true">通过</option><option value="false">失败</option></select></label><button onclick="downloadAll()">下载全部完整 JSON</button><button onclick="expandAll(true)">展开筛选结果</button><button onclick="expandAll(false)">全部收起</button><span id="count" aria-live="polite"></span></div>
<details><summary>运行元数据、模型 SHA256、容差与实际模块路径</summary><pre>METADATA</pre></details>
<p id="empty" hidden>没有匹配的用例，请调整名称或结果筛选。</p>CASES
<noscript>脚本已禁用：仍可展开全部用例查看完整数据；筛选和下载功能需要启用 JavaScript。</noscript>
<script id="full-data" type="application/json">PAYLOAD</script>
<script>
const full=JSON.parse(document.getElementById('full-data').textContent),cards=[...document.querySelectorAll('.case')];
function filter(){let q=document.getElementById('search').value.toLowerCase(),s=document.getElementById('filter').value,n=0;for(const c of cards){c.hidden=!(c.dataset.uid.toLowerCase().includes(q)&&(s==='all'||c.dataset.passed===s));if(!c.hidden)n++}document.getElementById('count').textContent=n+' / '+cards.length+' 个用例';document.getElementById('empty').hidden=n!==0}
function save(value,name){const url=URL.createObjectURL(new Blob([JSON.stringify(value,null,2)],{type:'application/json;charset=utf-8'}));const a=document.createElement('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)}
function downloadAll(){save(full,'mrmpformer-complete-results.json')}function downloadCase(i){save(full.cases[i],full.cases[i].summary.uid+'.json')}
function expandAll(open){for(const c of cards)if(!open||!c.hidden)c.open=open}
document.getElementById('search').addEventListener('input',filter);document.getElementById('filter').addEventListener('change',filter);filter();
</script></main></html>'''
    metrics = ''.join(f'<div><strong>{escape(str(v))}</strong>{label}</div>' for label, v in
                      [('用例', report['cases']), ('通过', report['passed']), ('两两对照', report['comparisons']), ('已通过用例最大差异', report['max_absolute_delta'])])
    html = html.replace('METRICS', metrics).replace('METADATA', escape(pretty(report))).replace('CASES', ''.join(body)).replace('PAYLOAD', data)
    target = output / 'report.html'
    target.write_text(html, encoding='utf-8')
    print(f'Complete HTML report: {target.resolve()} ({len(cases)} cases)')
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=Path(__file__).parent / 'results')
    render(parser.parse_args().results)
