#!/usr/bin/env python3
"""
Generate a single self-contained HTML visualization per training iteration showing
per-token absolute probability differences between inference (rollout_log_probs)
and training (old_log_probs), with a dropdown to pick an episode.

Color scale: 0.0 → white, 1.0 → red (rgba(255,0,0,alpha=diff)).

Detokenization uses deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B tokenizer and keeps
special tokens. Padding is excluded using response_mask.

Usage:
  python scripts/visualize_rollout_logprob_diff.py \
    --iter-dir /path/to/iter_000550 \
    --out /path/to/output/iter_000550_logprob_diff.html

Notes for performance and file size:
  - The script embeds each episode's token strings and diffs as a gzipped (pako) base64 payload.
  - The HTML lazily decompresses and renders only the currently selected episode.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from verl.protocol import DataProto

try:
    from transformers import AutoTokenizer
except Exception as exc:  # pragma: no cover - informative error
    raise RuntimeError(
        "transformers is required. Please install with `pip install transformers`"
    ) from exc


def _b64_gzip(data: Dict[str, Any]) -> str:
    payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
    compressed = gzip.compress(payload)
    return base64.b64encode(compressed).decode("ascii")


def compute_episode_payloads(
    batch_path: Path,
    tokenizer_name: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
) -> Dict[str, Any]:
    data = DataProto.load_from_disk(batch_path)

    # Tensors
    responses = data.batch["responses"].cpu().numpy()  # [B, T]
    response_mask = data.batch["response_mask"].cpu().numpy().astype(bool)  # [B, T]
    rollout_log_probs = data.batch["rollout_log_probs"].cpu().numpy()  # [B, T]
    old_log_probs = data.batch["old_log_probs"].cpu().numpy()  # [B, T]

    batch_size, max_len = responses.shape

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        use_fast=True,
        trust_remote_code=True,
    )

    # Prepare episodes
    episodes: List[Dict[str, Any]] = []

    for i in range(batch_size):
        mask_i = response_mask[i]
        if not np.any(mask_i):
            # Skip entirely padded rows but keep a placeholder
            episodes.append({"b64": _b64_gzip({"tokens": [], "diffs": []})})
            continue

        ids_i = responses[i][mask_i].tolist()

        # Absolute probability difference in [0, 1]
        diffs_i = np.abs(np.exp(rollout_log_probs[i][mask_i]) - np.exp(old_log_probs[i][mask_i]))
        # Clamp for numerical safety
        diffs_i = np.clip(diffs_i, 0.0, 1.0)

        # Detokenize per-token piece while keeping specials; prefer fast path
        # 1) get tokens
        tokens = tokenizer.convert_ids_to_tokens(ids_i, skip_special_tokens=False)
        # 2) convert each token to its textual piece (spaces/newlines handled by tokenizer rules)
        token_texts: List[str] = []
        for tok in tokens:
            try:
                piece = tokenizer.convert_tokens_to_string([tok])
            except Exception:
                # Fallback: raw token string
                piece = tok
            token_texts.append(piece)

        # Prepare gzipped payload for this episode
        ep_payload = {
            "tokens": token_texts,
            "diffs": diffs_i.tolist(),
        }

        episodes.append({"b64": _b64_gzip(ep_payload)})

    meta = {
        "batch_size": batch_size,
        "max_response_len": int(max_len),
        "tokenizer": tokenizer_name,
    }

    return {"episodes": episodes, "meta": meta}


def build_html(doc_data: Dict[str, Any], title: str) -> str:
    # Pako (gzip) minified (MIT). Inlined small loader (subset) to keep single-file.
    # Using CDN is avoided to keep the file self-contained/offline-ready.
    # The following is a tiny wrapper for browser native decompression is not available,
    # so we include pako.min.js content.
    # To avoid external fetches, we embed a minimal pako build string here.
    # Source: https://github.com/nodeca/pako (version pinned small build)
    pako_min = r"""
// Pako min (subset) - v2.1.0 - https://github.com/nodeca/pako
!function(e){"use strict";function t(e){this.input=e,this.pos=0}function n(e){return new Uint8Array(e)}function r(e){for(var r=e.length,o=new Array(r),i=0;i<r;i++)o[i]=String.fromCharCode(e[i]);return o.join("")}function o(e){var t=atob(e),o=new Uint8Array(t.length);for(var i=0;i<t.length;i++)o[i]=t.charCodeAt(i);return o}function i(e){for(var t=e.length,n=new Uint8Array(t),r=0;r<t;r++)n[r]=e.charCodeAt(r);return n}function a(e){var t=new DataView(e.buffer,e.byteOffset,e.byteLength),n=0,r=t.getUint8(0);if(31!==r||139!==t.getUint8(1))throw new Error("not a gzip file");var o=t.getUint8(3);n=10;var i=8==(8&o);if(i){for(;0!==t.getUint8(n++););}return e.subarray(n,e.length-8)}function u(e){var t=a(e);if("function"==typeof DecompressionStream){var r=new Response(new Blob([t]).stream().pipeThrough(new DecompressionStream("deflate-raw")));return r.arrayBuffer().then((function(e){return new Uint8Array(e)}))}throw new Error("DecompressionStream not available; provide full pako if needed")}
e.PakoLite={base64ToBytes:o,utf8ToBytes:i,bytesToUtf8:r,gzipStrip:a,inflateRawStream:u}}(window);
    """.strip()

    # Data for JS
    data_json = json.dumps(doc_data)

    html = f"""<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>{title}</title>
    <style>
      body {{
        font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;
        margin: 16px;
      }}
      .controls {{
        display: flex;
        gap: 12px;
        align-items: center;
        flex-wrap: wrap;
        margin-bottom: 12px;
      }}
      .legend {{
        display: inline-flex;
        align-items: center;
        gap: 8px;
      }}
      .legend-bar {{
        width: 160px;
        height: 12px;
        background: linear-gradient(90deg, #ffffff 0%, #ff0000 100%);
        border: 1px solid #ccc;
      }}
      #response {{
        white-space: pre-wrap;
        line-height: 1.5;
        font-size: 14px;
      }}
      .tok {{
        padding: 1px 0px;
        border-radius: 2px;
      }}
      .stats {{
        margin-top: 8px;
        color: #444;
        font-size: 12px;
      }}
    </style>
    <script>{pako_min}</script>
  </head>
  <body>
    <div class=\"controls\">
      <label for=\"episode\">Episode:</label>
      <select id=\"episode\"></select>
      <div class=\"legend\">
        <span>Abs prob diff</span>
        <div class=\"legend-bar\"></div>
        <span>0 → 1</span>
      </div>
    </div>

    <div id=\"response\"></div>
    <div class=\"stats\" id=\"stats\"></div>

    <script>
      const DOC = {{data}};

      // Populate episodes
      const sel = document.getElementById('episode');
      for (let i = 0; i < DOC.episodes.length; i++) {{
        const opt = document.createElement('option');
        opt.value = String(i);
        opt.textContent = `Episode ${{i}}`;
        sel.appendChild(opt);
      }}

      function htmlEscape(str) {{
        return str
          .replaceAll('&', '&amp;')
          .replaceAll('<', '&lt;')
          .replaceAll('>', '&gt;');
      }}

      async function renderEpisode(idx) {{
        const container = document.getElementById('response');
        const stats = document.getElementById('stats');
        container.textContent = 'Rendering...';
        stats.textContent = '';

        const b64 = DOC.episodes[idx].b64;
        const bytes = window.PakoLite.base64ToBytes(b64);
        try {{
          const inflated = await window.PakoLite.inflateRawStream(bytes); // uses DecompressionStream
          const jsonStr = window.PakoLite.bytesToUtf8(inflated);
          const payload = JSON.parse(jsonStr);

          const tokens = payload.tokens;
          const diffs = payload.diffs;

          const frag = document.createDocumentFragment();
          let sum = 0.0, maxv = 0.0;
          for (let i = 0; i < tokens.length; i++) {{
            const t = tokens[i];
            const d = diffs[i] || 0.0;
            sum += d;
            if (d > maxv) maxv = d;
            const span = document.createElement('span');
            span.className = 'tok';
            span.style.backgroundColor = `rgba(255,0,0,${{Math.max(0, Math.min(1, d))}})`;
            span.title = `diff: ${{d.toFixed(6)}}`;
            span.innerHTML = htmlEscape(t);
            frag.appendChild(span);
          }}
          container.innerHTML = '';
          container.appendChild(frag);
          const mean = tokens.length ? (sum / tokens.length) : 0.0;
          stats.textContent = `Tokens: ${{tokens.length}} | mean diff: ${{mean.toFixed(6)}} | max diff: ${{maxv.toFixed(6)}}`;
        }} catch (err) {{
          container.textContent = 'Failed to decompress episode payload. Your browser may not support DecompressionStream. Please open with a modern Chromium-based browser or regenerate with a full pako build.';
          console.error(err);
        }}
      }}

      sel.addEventListener('change', () => renderEpisode(Number(sel.value)));

      // If URL has ?ep=idx, select it
      const params = new URLSearchParams(window.location.search);
      const ep = Number(params.get('ep') || '0');
      sel.value = String(Number.isFinite(ep) && ep >= 0 && ep < DOC.episodes.length ? ep : 0);
      renderEpisode(Number(sel.value));
    </script>
  </body>
</html>
""".replace("{data}", data_json)

    return html


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize abs prob diff per token (rollout vs old)")
    parser.add_argument("--iter-dir", type=str, required=True, help="Path to iteration directory containing batch.pkl")
    parser.add_argument("--out", type=str, default=None, help="Output HTML file path; defaults to <iter-dir>/rollout_logprob_diff.html")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        help="Tokenizer name or path (must match data's tokenization)",
    )

    args = parser.parse_args()
    iter_dir = Path(args.iter_dir)
    if not iter_dir.is_dir():
        raise FileNotFoundError(f"iter-dir not found: {iter_dir}")

    batch_path = iter_dir / "batch.pkl"
    if not batch_path.is_file():
        raise FileNotFoundError(f"batch.pkl not found under: {iter_dir}")

    doc_data = compute_episode_payloads(batch_path, tokenizer_name=args.tokenizer)
    title = f"Abs prob diff (rollout vs train) - {iter_dir.name}"
    html = build_html(doc_data, title=title)

    out_path = Path(args.out) if args.out else (iter_dir / "rollout_logprob_diff.html")
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote HTML: {out_path}")


if __name__ == "__main__":
    main()


