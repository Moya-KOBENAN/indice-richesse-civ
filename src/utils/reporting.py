"""Export de rapports autonomes et traçabilité des analyses."""
from pathlib import Path
import base64
import hashlib
import html
import json
import platform
from datetime import datetime, timezone
from importlib.metadata import version


def sha256(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def prepare_output(path):
    path = Path(path)
    for folder in ['tables', 'figures']:
        (path / folder).mkdir(parents=True, exist_ok=True)
    return path


def save_table(frame, output, name):
    frame.to_csv(Path(output) / 'tables' / (name + '.csv'), index=False)


def save_figure(fig, output, name):
    import matplotlib.pyplot as plt
    fig.tight_layout()
    fig.savefig(Path(output) / 'figures' / (name + '.png'), dpi=160, bbox_inches='tight')
    plt.close(fig)


def render_report(output, title, sections, figures, sources=()):
    """Sections = titres, paragraphes et éventuels DataFrames déjà agrégés."""
    output = Path(output)
    html_parts = [f'<h1>{html.escape(title)}</h1>']
    markdown = ['# ' + title, '']
    for heading, paragraphs, table in sections:
        html_parts.append(f'<h2>{html.escape(heading)}</h2>')
        markdown += ['## ' + heading, '']
        for paragraph in paragraphs:
            html_parts.append('<p>' + html.escape(paragraph) + '</p>')
            markdown += [paragraph, '']
        if table is not None:
            html_parts.append(table.to_html(index=False, border=0, escape=True, float_format=lambda v: f'{v:.4f}'))
            markdown += ['```text', table.to_string(index=False), '```', '']
    for filename, caption in figures:
        image = output / 'figures' / (filename + '.png')
        encoded = base64.b64encode(image.read_bytes()).decode('ascii')
        html_parts.append(f'<figure><img src="data:image/png;base64,{encoded}" alt="{html.escape(caption)}"><figcaption>{html.escape(caption)}</figcaption></figure>')
        markdown += [f'![{caption}](figures/{filename}.png)', '']
    if sources:
        html_parts.append('<h2>Sources documentaires</h2><ul>')
        markdown += ['## Sources documentaires', '']
        for label, url in sources:
            html_parts.append(f'<li><a href="{html.escape(url)}">{html.escape(label)}</a></li>')
            markdown += [f'- [{label}]({url})']
        html_parts.append('</ul>')
    style = 'body{font:16px/1.6 system-ui,sans-serif;max-width:1080px;margin:40px auto;padding:0 24px;color:#243348}h1,h2{color:#173d5a}table{border-collapse:collapse;font-size:14px;display:block;overflow:auto}td,th{padding:8px 12px;border-bottom:1px solid #ccd6df;text-align:left}img{max-width:100%}figure{margin:32px 0}figcaption{color:#526172}a{color:#126485}'
    (output/'rapport.html').write_text('<!doctype html><html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+'</title><style>'+style+'</style><body>'+''.join(html_parts)+'</body></html>')
    (output/'RAPPORT_EDA.md').write_text('\n'.join(markdown)+'\n')


def write_manifest(output, inputs, script, scope, extra=None):
    sources = {str(Path(p).resolve()): sha256(p) for p in inputs}
    code = {str(Path(script).resolve()): sha256(script)}
    for name in ['statistics.py', 'reporting.py']:
        path = Path(__file__).with_name(name)
        code[str(path)] = sha256(path)
    artifacts = {str(p.relative_to(output)): sha256(p) for p in Path(output).rglob('*') if p.is_file() and p.name != 'manifest.json'}
    data = {'scope': scope, 'created_utc': datetime.now(timezone.utc).isoformat(),
            'python': platform.python_version(), 'packages': {p: version(p) for p in ['numpy','pandas','matplotlib']},
            'input_sha256': sources, 'code_sha256': code, 'artifact_sha256': artifacts,
            'status': 'EDA descriptive initiale ; aucun modèle ajusté', **(extra or {})}
    (Path(output)/'manifest.json').write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n')
