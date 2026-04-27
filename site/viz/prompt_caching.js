// viz/prompt_caching.js
// Latency + cost reduction from prompt caching (Anthropic/OpenAI).

registerViz('prompt_caching', function (container) {
  container.innerHTML = `
    <p class="viz-title">Prompt caching in production</p>
    <p class="viz-sub">Big system prompt + RAG context? Cache the common prefix. Subsequent calls skip prefill for cached tokens.</p>
    <div class="viz-controls">
      <label>Prefix tokens: <input id="pc-p" type="range" min="0" max="100000" step="500" value="20000"><span id="pc-p-out">20,000</span></label>
      <label>Query tokens: <input id="pc-q" type="range" min="50" max="2000" step="50" value="200"><span id="pc-q-out">200</span></label>
      <label>Hit rate: <input id="pc-h" type="range" min="0" max="100" value="90"><span id="pc-h-out">90%</span></label>
    </div>
    <div style="display:flex;gap:12px;flex-wrap:wrap;margin-top:12px">
      <div style="flex:1;min-width:220px;background:#fee2e2;border-radius:6px;padding:12px">
        <div style="font-weight:bold;color:#991b1b;margin-bottom:6px">Without caching</div>
        <div id="pc-no" style="font-family:ui-monospace,monospace;font-size:12px;line-height:1.8"></div>
      </div>
      <div style="flex:1;min-width:220px;background:#d1fae5;border-radius:6px;padding:12px">
        <div style="font-weight:bold;color:#065f46;margin-bottom:6px">With caching</div>
        <div id="pc-yes" style="font-family:ui-monospace,monospace;font-size:12px;line-height:1.8"></div>
      </div>
    </div>
    <p class="viz-readout" id="pc-out"></p>
  `;

  function render() {
    const P = +container.querySelector('#pc-p').value;
    const Q = +container.querySelector('#pc-q').value;
    const H = +container.querySelector('#pc-h').value / 100;
    container.querySelector('#pc-p-out').textContent = P.toLocaleString();
    container.querySelector('#pc-q-out').textContent = Q.toLocaleString();
    container.querySelector('#pc-h-out').textContent = (H * 100).toFixed(0) + '%';

    // costs (per-1M tokens, rough Claude Sonnet 3.5 rates)
    const INPUT = 3.00;
    const OUTPUT = 15.00;
    const CACHE_WRITE = 3.75;   // 1.25x input
    const CACHE_READ = 0.30;    // 0.1x input
    const OUT_TOKENS = 500;
    // Without caching: 100 calls
    const calls = 100;
    const noIn = calls * (P + Q) * INPUT / 1e6;
    const noOut = calls * OUT_TOKENS * OUTPUT / 1e6;
    const noTotal = noIn + noOut;
    // With caching: 1 write, 99 reads weighted by hit rate
    const writes = calls * (1 - H);
    const reads = calls * H;
    const wIn = writes * (P + Q) * INPUT / 1e6 + reads * Q * INPUT / 1e6 + writes * P * (CACHE_WRITE - INPUT) / 1e6 + reads * P * CACHE_READ / 1e6;
    const wOut = calls * OUT_TOKENS * OUTPUT / 1e6;
    const wTotal = wIn + wOut;
    const savings = ((1 - wTotal / noTotal) * 100).toFixed(0);
    // Latency (rough: 1ms per prefill token, 10ms per output)
    const noLat = (P + Q) * 1 + OUT_TOKENS * 10;
    const wLatHit = Q * 1 + OUT_TOKENS * 10; // prefix skipped
    const wLatMiss = (P + Q) * 1 + OUT_TOKENS * 10;
    const wLat = H * wLatHit + (1 - H) * wLatMiss;

    container.querySelector('#pc-no').innerHTML =
      `Prefill tokens: <b>${((P+Q)*calls/1e6).toFixed(2)}M</b><br>` +
      `Input cost: <b>$${noIn.toFixed(2)}</b><br>` +
      `Output cost: $${noOut.toFixed(2)}<br>` +
      `<b>Total: $${noTotal.toFixed(2)}</b> / 100 calls<br>` +
      `p50 latency: <b>${(noLat/1000).toFixed(1)}s</b>`;
    container.querySelector('#pc-yes').innerHTML =
      `Cache writes: ${writes.toFixed(0)}, reads: ${reads.toFixed(0)}<br>` +
      `Input+cache cost: <b>$${wIn.toFixed(2)}</b><br>` +
      `Output cost: $${wOut.toFixed(2)}<br>` +
      `<b>Total: $${wTotal.toFixed(2)}</b> / 100 calls<br>` +
      `p50 latency (hit): <b>${(wLatHit/1000).toFixed(1)}s</b>`;
    container.querySelector('#pc-out').innerHTML =
      `<b>${savings}% cheaper</b>. Latency drops ${(noLat/wLatHit).toFixed(1)}× on cache hit. ` +
      `Biggest wins: large system prompts, RAG contexts, few-shot examples. ` +
      `TTL typically 5 min (Anthropic) — cache refreshes on reuse.`;
  }
  container.querySelectorAll('input').forEach(el => el.addEventListener('input', render));
  render();
});
