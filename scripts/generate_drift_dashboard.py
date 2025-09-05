#!/usr/bin/env python3
"""
Generate an interactive single-file HTML dashboard visualizing logprob drift across iterations.

Visualizations:
- Time series: per-iteration mean abs prob diff
- Global histogram: distribution of abs diffs across all iterations
- Top tokens: bar chart switchable by mean/total/high-rate; searchable token table

Data pipeline:
- Scans a root experiments directory for `iter_*` subdirectories with `batch.pkl`
- Uses `response_mask` to filter padding, computes |p_rollout - p_train| with both probs in [0,1]
- Aggregates per-token stats (count, sum, max, high_count) and per-iteration metrics
- Detokenizes with deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B, keeping specials

The resulting HTML embeds gzipped+base64 JSON payload. Charts render in-browser.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from transformers import AutoTokenizer

from verl.protocol import DataProto


@dataclass
class TokenStats:
    token: str
    piece: str
    count: int
    diff_sum: float
    high_count: int
    max_diff: float
    is_special: bool

    @property
    def mean_diff(self) -> float:
        return (self.diff_sum / self.count) if self.count > 0 else 0.0

    @property
    def high_rate(self) -> float:
        return (self.high_count / self.count) if self.count > 0 else 0.0


def find_iteration_dirs(root_dir: Path) -> List[Path]:
    return sorted([p for p in root_dir.iterdir() if p.is_dir() and p.name.startswith("iter_")])


def parse_step(iter_dir: Path) -> int:
    # iter_000350 -> 350
    try:
        return int(iter_dir.name.split("_")[-1].lstrip("0") or "0")
    except Exception:
        return 0


def _b64_gzip_payload(obj: Dict[str, Any]) -> str:
    raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    comp = gzip.compress(raw)
    return base64.b64encode(comp).decode("ascii")


def update_histogram(hist: np.ndarray, diffs: np.ndarray) -> None:
    num_bins = hist.shape[0]
    clipped = np.clip(diffs, 0.0, 1.0)
    idx = np.floor(clipped * num_bins).astype(np.int64)
    np.minimum(idx, num_bins - 1, out=idx)
    np.maximum(idx, 0, out=idx)
    binc = np.bincount(idx.ravel(), minlength=num_bins)
    hist += binc[: num_bins]


def analyze(
    root_dir: Path,
    tokenizer_name: str,
    high_threshold: float,
    num_bins: int,
    limit_iters: int | None,
) -> Tuple[Dict[str, TokenStats], List[Dict[str, Any]], Dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        trust_remote_code=True,
    )
    special_tokens_set = set(tokenizer.all_special_tokens)

    token_to_stats: Dict[str, TokenStats] = {}
    iter_metrics: List[Dict[str, Any]] = []
    global_hist = np.zeros((num_bins,), dtype=np.int64)

    iters = find_iteration_dirs(root_dir)
    if limit_iters is not None:
        iters = iters[:limit_iters]

    for iter_dir in iters:
        batch_path = iter_dir / "batch.pkl"
        if not batch_path.is_file():
            continue

        data = DataProto.load_from_disk(batch_path)
        responses = data.batch["responses"].cpu().numpy()  # [B, T]
        response_mask = data.batch["response_mask"].cpu().numpy().astype(bool)  # [B, T]
        rollout_log_probs = data.batch["rollout_log_probs"].cpu().numpy()  # [B, T]
        old_log_probs = data.batch["old_log_probs"].cpu().numpy()  # [B, T]

        if not response_mask.any():
            continue

        prob_rollout_all = np.exp(rollout_log_probs[response_mask])
        prob_old_all = np.exp(old_log_probs[response_mask])
        diffs_all = np.abs(prob_rollout_all - prob_old_all)
        diffs_all = np.clip(diffs_all, 0.0, 1.0)

        # Per-iteration metrics
        mean_i = float(diffs_all.mean()) if diffs_all.size else 0.0
        max_i = float(diffs_all.max()) if diffs_all.size else 0.0
        high_rate_i = float((diffs_all >= high_threshold).mean()) if diffs_all.size else 0.0
        iter_metrics.append(
            {
                "name": iter_dir.name,
                "step": parse_step(iter_dir),
                "mean": mean_i,
                "max": max_i,
                "high_rate": high_rate_i,
            }
        )

        # Global histogram
        update_histogram(global_hist, diffs_all)

        # Token-wise stats
        B, _ = responses.shape
        for i in range(B):
            mask_i = response_mask[i]
            if not mask_i.any():
                continue
            ids_i = responses[i][mask_i].tolist()
            diffs_i = np.abs(np.exp(rollout_log_probs[i][mask_i]) - np.exp(old_log_probs[i][mask_i]))
            diffs_i = np.clip(diffs_i, 0.0, 1.0)
            toks_i = tokenizer.convert_ids_to_tokens(ids_i, skip_special_tokens=False)

            # Convert to pieces (visible strings)
            pieces_i: List[str] = []
            for tok in toks_i:
                try:
                    piece = tokenizer.convert_tokens_to_string([tok])
                except Exception:
                    piece = tok
                pieces_i.append(piece)

            for tok, piece, d in zip(toks_i, pieces_i, diffs_i.tolist()):
                st = token_to_stats.get(tok)
                if st is None:
                    st = TokenStats(
                        token=tok,
                        piece=piece,
                        count=0,
                        diff_sum=0.0,
                        high_count=0,
                        max_diff=0.0,
                        is_special=(tok in special_tokens_set),
                    )
                    token_to_stats[tok] = st
                st.count += 1
                st.diff_sum += float(d)
                if d >= high_threshold:
                    st.high_count += 1
                if d > st.max_diff:
                    st.max_diff = float(d)

    meta = {
        "tokenizer": tokenizer_name,
        "high_threshold": high_threshold,
        "num_bins": num_bins,
    }
    return token_to_stats, iter_metrics, {"hist_counts": global_hist.tolist(), **meta}


def build_html(
    token_to_stats: Dict[str, TokenStats],
    iter_metrics: List[Dict[str, Any]],
    global_info: Dict[str, Any],
    min_count: int,
    max_tokens: int,
    title: str,
) -> str:
    # Prepare token list with filtering and truncation for payload size
    tokens_all = [
        {
            "token": k,
            "piece": v.piece,
            "count": v.count,
            "mean": v.mean_diff,
            "total": v.diff_sum,
            "high_rate": v.high_rate,
            "max": v.max_diff,
            "special": v.is_special,
        }
        for k, v in token_to_stats.items()
        if v.count >= min_count
    ]
    # Keep top by total to bound payload
    tokens_all.sort(key=lambda o: o["total"], reverse=True)
    tokens_all = tokens_all[:max_tokens]

    payload = {
        "global": global_info,
        "iterations": sorted(iter_metrics, key=lambda m: m["step"]),
        "tokens": tokens_all,
    }
    b64 = _b64_gzip_payload(payload)

    # Embed Chart.js v4.4.1 minified (license MIT). For brevity, not the full source here.
    # If needed, we can swap to a smaller chart helper.
    chart_js = r"""
/* Chart.js v4 minimal build placeholder. For production, replace with full minified source if needed. */
""".strip()

    html = f"""<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>{title}</title>
    <style>
      body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Ubuntu, Cantarell, Noto Sans, Arial, 'Apple Color Emoji', 'Segoe UI Emoji'; margin: 16px; }}
      h2 {{ margin: 16px 0 8px; }}
      .grid {{ display: grid; grid-template-columns: 1fr; gap: 16px; }}
      @media (min-width: 1100px) {{ .grid {{ grid-template-columns: 1fr 1fr; }} }}
      .card {{ border: 1px solid #e3e3e3; border-radius: 8px; padding: 12px; background: #fff; }}
      .row {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; }}
      .small {{ color: #555; font-size: 12px; }}
      .table {{ max-height: 500px; overflow: auto; border: 1px solid #eee; }}
      table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
      th, td {{ border-bottom: 1px solid #f0f0f0; padding: 6px 8px; text-align: left; }}
      tr:hover {{ background: #fafafa; }}
      input[type='search'] {{ padding: 6px 8px; width: 280px; }}
      select {{ padding: 4px 6px; }}
      .legend-bar {{ width: 160px; height: 12px; background: linear-gradient(90deg, #ffffff 0%, #ff0000 100%); border: 1px solid #ccc; }}
    </style>
    <script>{chart_js}</script>
  </head>
  <body>
    <h2>Logprob drift dashboard</h2>
    <div class="row small">
      <div>Tokenizer: <code>{global_info['tokenizer']}</code></div>
      <div>High-diff threshold: <code>{global_info['high_threshold']}</code></div>
      <div class="row" style="gap:8px; align-items:center;">
        <div>Color scale (0→1):</div>
        <div class="legend-bar"></div>
      </div>
    </div>

    <div class="grid">
      <div class="card">
        <h3>Per-iteration mean abs diff</h3>
        <canvas id="ts"></canvas>
      </div>
      <div class="card">
        <h3>Global distribution (histogram)</h3>
        <canvas id="hist"></canvas>
      </div>
    </div>

    <div class="card" style="margin-top:16px;">
      <div class="row">
        <h3 style="margin-right:12px;">Top tokens</h3>
        <label>Metric:
          <select id="metric">
            <option value="total">total</option>
            <option value="mean">mean</option>
            <option value="high_rate">high_rate</option>
            <option value="max">max</option>
          </select>
        </label>
        <label>Top-K: <input type="number" id="topk" value="50" min="5" max="2000" step="5" style="width:80px;"/></label>
      </div>
      <canvas id="topTokens" style="max-height:480px;"></canvas>
    </div>

    <div class="card" style="margin-top:16px;">
      <div class="row">
        <h3 style="margin-right:12px;">Token search</h3>
        <input id="search" type="search" placeholder="substring in token or piece..." />
      </div>
      <div class="table">
        <table id="tokTable">
          <thead>
            <tr><th>token</th><th>piece</th><th>count</th><th>mean</th><th>total</th><th>high_rate</th><th>max</th><th>special</th></tr>
          </thead>
          <tbody></tbody>
        </table>
      </div>
    </div>

    <script>
      const B64 = "{b64}";
      async function loadPayload() {{
        const compressed = Uint8Array.from(atob(B64), c => c.charCodeAt(0));
        if (typeof DecompressionStream === 'function') {{
          const ds = new DecompressionStream('gzip');
          const decompressed = await new Response(new Blob([compressed]).stream().pipeThrough(ds)).arrayBuffer();
          return JSON.parse(new TextDecoder().decode(decompressed));
        }} else {{
          // Fallback: try inflate-raw after stripping gzip header via custom small routine
          alert('Browser lacks DecompressionStream. Please use a modern Chromium-based browser.');
          return {{ global: {{}}, iterations: [], tokens: [] }};
        }}
      }}

      function fmt(x, k=6) {{ return Number(x).toFixed(k); }}

      function renderTable(rows) {{
        const tbody = document.querySelector('#tokTable tbody');
        tbody.innerHTML = '';
        for (const r of rows) {{
          const tr = document.createElement('tr');
          tr.innerHTML = `<td><code>${{r.token}}</code></td><td>${{r.piece.replaceAll('<','&lt;')}}</td><td>${{r.count}}</td>` +
            `<td>${{fmt(r.mean,6)}}</td><td>${{fmt(r.total,3)}}</td><td>${{fmt(r.high_rate,3)}}</td><td>${{fmt(r.max,6)}}</td><td>${{r.special}}</td>`;
          tbody.appendChild(tr);
        }}
      }}

      function makeLine(ctx, xs, ys, label) {{
        // Minimal canvas line chart
        const W = ctx.canvas.clientWidth; const H = 300; ctx.canvas.height = H; ctx.clearRect(0,0,ctx.canvas.width,ctx.canvas.height);
        const pad = 30; const x0 = pad, x1 = ctx.canvas.width - pad; const y0 = H - pad, y1 = pad;
        const minY = Math.min(...ys); const maxY = Math.max(...ys);
        const minX = Math.min(...xs); const maxX = Math.max(...xs);
        const sx = (v) => x0 + (x1 - x0) * (v - minX) / Math.max(1e-12, (maxX - minX));
        const sy = (v) => y0 - (y0 - y1) * (v - minY) / Math.max(1e-12, (maxY - minY));
        ctx.strokeStyle = '#1f77b4'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(sx(xs[0]), sy(ys[0]));
        for (let i = 1; i < xs.length; i++) {{ ctx.lineTo(sx(xs[i]), sy(ys[i])); }} ctx.stroke();
        ctx.fillStyle = '#333'; ctx.font = '12px sans-serif'; ctx.fillText(label, x0, y1 - 8);
      }}

      function makeBars(ctx, labels, values, horiz=false) {{
        const W = ctx.canvas.clientWidth; const H = Math.min(40 + values.length * 18, 600); ctx.canvas.height = H; ctx.clearRect(0,0,ctx.canvas.width,ctx.canvas.height);
        const pad = 60; const x0 = pad, x1 = ctx.canvas.width - 10; const y0 = H - 10, y1 = 10;
        const maxV = Math.max(...values, 1e-6);
        ctx.fillStyle = '#eaeaea'; ctx.fillRect(0,0,ctx.canvas.width,ctx.canvas.height);
        ctx.fillStyle = '#d62728';
        const barH = Math.max(12, Math.min(20, (H - 20) / values.length - 4));
        for (let i = 0; i < values.length; i++) {{
          const v = values[i];
          const w = (x1 - x0) * (v / maxV);
          const y = y1 + i * (barH + 4);
          ctx.fillRect(x0, y, w, barH);
        }}
        ctx.fillStyle = '#333'; ctx.font = '11px monospace';
        for (let i = 0; i < values.length; i++) {{
          const label = labels[i];
          const v = values[i];
          const y = y1 + i * (barH + 4);
          ctx.fillText(String(label).slice(0,60), 4, y + Math.min(barH,12));
          ctx.fillText(fmt(v, 6), x0 + 4, y + Math.min(barH,12));
        }}
      }}

      (async function main() {{
        const DATA = await loadPayload();
        const iters = DATA.iterations;
        const tokens = DATA.tokens;
        // Time series
        const xs = iters.map(r => r.step); const ys = iters.map(r => r.mean);
        makeLine(document.getElementById('ts').getContext('2d'), xs, ys, 'mean abs diff');
        // Histogram
        const hist = DATA.global.hist_counts; const labels = hist.map((_,i) => i / Math.max(1, hist.length-1));
        makeBars(document.getElementById('hist').getContext('2d'), labels.map(x => x.toFixed(2)), hist.slice(0, Math.min(100, hist.length)));
        // Top tokens chart
        function renderTop() {{
          const metric = document.getElementById('metric').value; const k = Math.max(5, Math.min(2000, Number(document.getElementById('topk').value)||50));
          const sorted = tokens.slice().sort((a,b) => (b[metric] - a[metric])).slice(0, k);
          const labs = sorted.map(o => o.piece || o.token); const vals = sorted.map(o => o[metric]);
          makeBars(document.getElementById('topTokens').getContext('2d'), labs, vals);
        }}
        document.getElementById('metric').addEventListener('change', renderTop);
        document.getElementById('topk').addEventListener('change', renderTop);
        renderTop();
        // Table + search
        function applySearch() {{
          const q = (document.getElementById('search').value||'').toLowerCase();
          const rows = q ? tokens.filter(o => (o.token.toLowerCase().includes(q) || (o.piece||'').toLowerCase().includes(q))) : tokens.slice(0, 500);
          renderTable(rows);
        }}
        document.getElementById('search').addEventListener('input', applySearch);
        applySearch();
      }})();
    </script>
  </body>
</html>
"""
    return html


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate interactive drift dashboard as a single HTML file")
    parser.add_argument("--root-dir", type=str, required=True, help="Root experiments directory with iter_* subdirs")
    parser.add_argument("--out", type=str, required=True, help="Path to output HTML file")
    parser.add_argument("--tokenizer", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    parser.add_argument("--high-threshold", type=float, default=0.2)
    parser.add_argument("--num-bins", type=int, default=1000)
    parser.add_argument("--min-count", type=int, default=100, help="Min token count to include in payload")
    parser.add_argument("--max-tokens", type=int, default=5000, help="Max tokens to embed (top by total)")
    parser.add_argument("--limit-iters", type=int, default=None, help="Optional: limit number of iterations")

    args = parser.parse_args()
    root_dir = Path(args.root_dir)
    out_path = Path(args.out)

    token_to_stats, iter_metrics, global_info = analyze(
        root_dir=root_dir,
        tokenizer_name=args.tokenizer,
        high_threshold=args.high_threshold,
        num_bins=args.num_bins,
        limit_iters=args.limit_iters,
    )

    title = f"Drift dashboard - {root_dir.name}"
    html = build_html(
        token_to_stats=token_to_stats,
        iter_metrics=iter_metrics,
        global_info=global_info,
        min_count=args.min_count,
        max_tokens=args.max_tokens,
        title=title,
    )
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote HTML: {out_path}")


if __name__ == "__main__":
    main()


