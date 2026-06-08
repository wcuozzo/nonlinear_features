"""
Local web dashboard for the loss-vs-config phase diagrams.

Serves all 6 choose-2 plot groups using Plotly heatmaps. Server reads CSV
on each request, regenerates plots only if CSV mtime changed.
Manual refresh in browser — no auto-refresh.

Usage:
    python dashboard.py  # then open http://127.0.0.1:5050
"""
import argparse
import os
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from flask import Flask, render_template_string


app = Flask(__name__)
CSV_PATH = Path('results_db/compiled/sweep_results_precise.csv')

# Cache: (csv_mtime, html_fragment)
_cache = {'mtime': None}


def load_df():
    return pd.read_csv(CSV_PATH)


def build_heatmap_subgroup(df, fixed, varying, vmin, vmax):
    """Returns a Plotly figure: a grid of small-multiple heatmaps for one fixed pair.

    Each panel:
      - Drops rows/cols that are entirely empty (so panels can have different
        sizes — no wasted space).
      - Empty cells (missing config) have no hover, just blank background.
    """
    a, b = fixed
    c, d = varying
    a_vals = sorted(df[a].unique())
    b_vals = sorted(df[b].unique())

    n_rows = len(a_vals)
    n_cols = len(b_vals)
    # Plenty of vertical space to avoid title overlap
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        subplot_titles=[f'{a}={av} {b}={bv}'
                       for av in a_vals for bv in b_vals],
        horizontal_spacing=0.05,
        vertical_spacing=0.10,
    )

    for i, av in enumerate(a_vals):
        for j, bv in enumerate(b_vals):
            sub = df[(df[a] == av) & (df[b] == bv)]
            if len(sub) == 0:
                # Add an empty annotation
                continue
            pivot = sub.pivot_table(values='mse_full', index=c, columns=d, aggfunc='first')
            # Drop rows/cols that are entirely NaN
            pivot = pivot.dropna(how='all', axis=0).dropna(how='all', axis=1)
            if pivot.shape[0] == 0 or pivot.shape[1] == 0:
                continue
            # Sort
            pivot = pivot.sort_index().sort_index(axis=1)

            log_z = np.log10(pivot.values)
            # Build text grid (empty string where NaN)
            text = [[(f'{v:.5f}' if not np.isnan(v) else '') for v in row]
                    for row in pivot.values]
            # Custom hover that handles NaN
            customdata = [[('missing' if np.isnan(v) else f'{v:.6f}') for v in row]
                          for row in pivot.values]
            heat = go.Heatmap(
                z=log_z,
                x=[str(v) for v in pivot.columns],
                y=[str(v) for v in pivot.index],
                text=text,
                texttemplate='%{text}',
                textfont=dict(size=10, color='white'),
                customdata=customdata,
                hovertemplate=(
                    f'{c}=%{{y}}<br>{d}=%{{x}}<br>MSE=%{{customdata}}'
                    f'<extra>{a}={av} {b}={bv}</extra>'
                ),
                colorscale='Viridis',
                zmin=np.log10(vmin), zmax=np.log10(vmax),
                showscale=False,
            )
            fig.add_trace(heat, row=i + 1, col=j + 1)
            fig.update_xaxes(title_text=d, row=i + 1, col=j + 1,
                             title_font=dict(size=10),
                             tickfont=dict(size=9))
            fig.update_yaxes(title_text=c, row=i + 1, col=j + 1,
                             title_font=dict(size=10),
                             tickfont=dict(size=9))

    # Add a single colorbar (off to the right)
    fig.add_trace(go.Heatmap(
        z=[[np.log10(vmin), np.log10(vmax)]],
        showscale=True, colorscale='Viridis',
        zmin=np.log10(vmin), zmax=np.log10(vmax),
        opacity=0,
        colorbar=dict(title='log10 MSE', x=1.04, len=0.7),
        hoverinfo='skip',
    ), row=1, col=1)

    # Generous height so subplot titles don't crowd
    row_h = 240  # per-row height
    fig.update_layout(
        title=f'MSE({c}, {d}) at fixed ({a}, {b})',
        height=row_h * n_rows + 120,
        width=200 * n_cols + 200,
        margin=dict(l=50, r=120, t=80, b=60),
        font=dict(size=10),
        plot_bgcolor='white',
    )
    return fig


TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset='utf-8'>
  <title>Loss Phase Diagrams Dashboard</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 1em 2em; background:#fafafa; color:#222;}
    h1 { margin-bottom: 0; }
    h2 { border-top: 1px solid #ccc; padding-top: 1em; margin-top: 2em; }
    .meta { color: #666; font-size: 12px; margin-bottom: 1em; }
    .summary { background: #eef; padding: 0.6em 1em; border-radius: 4px; font-size: 13px; margin-bottom: 1em; }
    .selector { background: #fff; padding: 1em; border: 1px solid #ccc; border-radius: 6px; margin-bottom: 1em; }
    .selector label { display: inline-block; padding: 6px 12px; margin: 4px;
                      background: #eee; border-radius: 4px; cursor: pointer; user-select: none; }
    .selector input[type="radio"] { display: none; }
    .selector input[type="radio"]:checked + span { background: #36a; color: white; padding: 6px 12px;
                                                    border-radius: 4px; }
    .selector input[type="radio"]:not(:checked) + span { padding: 6px 12px; }
    .plot { background: white; padding: 0.5em; border: 1px solid #ddd; border-radius: 4px; margin-bottom: 2em; }
    .panel { display: none; }
    .panel.active { display: block; }
    .refresh { padding: 6px 12px; background:#4a7; color:white; text-decoration: none; border-radius: 4px; font-size: 12px;}
  </style>
</head>
<body>
  <h1>Nonlinear-feature autoencoder · loss phase diagrams</h1>
  <div class='meta'>
    last server-side regen: {{generated_at}} · CSV mtime: {{csv_mtime}}<br>
    source: <code>{{csv_path}}</code> ({{n_configs}} configs) ·
    <a class='refresh' href='/' >Refresh page</a>
  </div>
  <div class='summary'>{{summary | safe}}</div>

  <div class='selector'>
    <b>View slicing:</b> (each shows all 216 configs, sliced differently)<br>
    {% for fixed_pair, varying_pair, plot in plots %}
      <label>
        <input type='radio' name='view' value='panel-{{loop.index0}}'
               {% if loop.index0 == default_idx %}checked{% endif %}
               onchange='switchPanel(this.value)'>
        <span>fix ({{fixed_pair[0]}}, {{fixed_pair[1]}}) → vary ({{varying_pair[0]}}, {{varying_pair[1]}})</span>
      </label>
    {% endfor %}
  </div>

  {% for fixed_pair, varying_pair, plot in plots %}
    <div id='panel-{{loop.index0}}' class='panel {% if loop.index0 == default_idx %}active{% endif %}'>
      <h2>Fix ({{fixed_pair[0]}}, {{fixed_pair[1]}}) → vary ({{varying_pair[0]}}, {{varying_pair[1]}})</h2>
      <div class='plot'>{{ plot | safe }}</div>
    </div>
  {% endfor %}

  <script>
    function switchPanel(id) {
      document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
      var el = document.getElementById(id);
      if (el) el.classList.add('active');
      // Plotly heatmaps don't resize correctly when revealed; force redraw
      setTimeout(function() {
        if (el) {
          var plots = el.querySelectorAll('.js-plotly-plot');
          plots.forEach(function(p) { if (window.Plotly) Plotly.Plots.resize(p); });
        }
      }, 50);
    }
  </script>
</body>
</html>
"""


@app.route('/')
def index():
    if not CSV_PATH.exists():
        return f'CSV not found: {CSV_PATH}', 500
    mtime = os.path.getmtime(CSV_PATH)
    if _cache.get('mtime') != mtime:
        df = load_df()
        all_axes = ['n', 'm', 'l', 'S']
        plots = []
        vmin = float(df.mse_full.min()) * 0.9
        vmax = float(df.mse_full.max()) * 1.1
        for fixed in combinations(all_axes, 2):
            varying = tuple(a for a in all_axes if a not in fixed)
            fig = build_heatmap_subgroup(df, fixed, varying, vmin, vmax)
            include = True if not plots else False
            plots.append((fixed, varying,
                          pio.to_html(fig, full_html=False, include_plotlyjs=include)))
        summary = (f'MSE range: <b>{df.mse_full.min():.6f}</b> to <b>{df.mse_full.max():.6f}</b> · '
                   f'l=1: {(df.l == 1).sum()} configs · l=2: {(df.l == 2).sum()} · '
                   f'l=3: {(df.l == 3).sum()} · l=4: {(df.l == 4).sum()}')
        _cache['mtime'] = mtime
        _cache['plots'] = plots
        _cache['summary'] = summary
        _cache['n_configs'] = len(df)
        _cache['generated_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    # Find default panel: fix (n,m) → vary (l,S)
    default_idx = 0
    for i, (fixed, varying, _) in enumerate(_cache['plots']):
        if fixed == ('n', 'm') and varying == ('l', 'S'):
            default_idx = i
            break

    return render_template_string(
        TEMPLATE,
        plots=_cache['plots'],
        summary=_cache['summary'],
        n_configs=_cache['n_configs'],
        generated_at=_cache['generated_at'],
        csv_mtime=time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(_cache['mtime'])),
        csv_path=str(CSV_PATH),
        default_idx=default_idx,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5050)
    parser.add_argument('--host', default='127.0.0.1')
    args = parser.parse_args()
    print(f'\nDashboard at http://{args.host}:{args.port}')
    print(f'CSV source: {CSV_PATH.resolve()}')
    print('Manual refresh in browser. Server regenerates only if CSV mtime changes.\n')
    app.run(host=args.host, port=args.port, debug=False)
