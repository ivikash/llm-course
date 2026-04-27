// viz/kv_cache.js
// Animated: see how KV cache turns O(T^2) generation into O(T) per token.

registerViz('kv_cache', function (container) {
  container.innerHTML = `
    <p class="viz-title">KV cache: why generation isn't quadratic</p>
    <p class="viz-sub">Each cell = one K/V vector for one token position. Without cache (left): recompute ALL past K/V every step. With cache (right): reuse past K/V, compute only the new one. Press Generate.</p>
    <div class="viz-controls">
      <label>Tokens to generate: <input id="kv-n" type="range" min="3" max="16" value="8"><span id="kv-n-out">8</span></label>
      <button id="kv-play">Generate</button>
      <button id="kv-reset">Reset</button>
    </div>
    <div style="display:flex;gap:24px;justify-content:center;flex-wrap:wrap">
      <div>
        <div style="font-size:12px;color:#6b7280;margin-bottom:4px;text-align:center">Without KV cache</div>
        <svg id="kv-no" viewBox="0 0 300 300" width="300" height="300" style="background:#fff;border:1px solid #e5e7eb;border-radius:4px"></svg>
      </div>
      <div>
        <div style="font-size:12px;color:#6b7280;margin-bottom:4px;text-align:center">With KV cache</div>
        <svg id="kv-yes" viewBox="0 0 300 300" width="300" height="300" style="background:#fff;border:1px solid #e5e7eb;border-radius:4px"></svg>
      </div>
    </div>
    <p class="viz-readout" id="kv-out">Without cache: ~T² work. With cache: ~T work. This is why modern LLMs generate tokens in O(1) per step, not O(T).</p>
  `;
  let step = 0;
  let playing = false;
  function render() {
    const N = +container.querySelector('#kv-n').value;
    container.querySelector('#kv-n-out').textContent = N;
    const size = Math.floor(260 / N);
    function draw(elId, withCache) {
      let html = '';
      let count = 0;
      for (let t = 0; t <= Math.min(step, N - 1); t++) {
        for (let p = 0; p <= t; p++) {
          // In without-cache: at step t, recompute all positions 0..t
          // In with-cache: at step t, only compute position t (others are cached)
          const isCurrent = withCache ? (p === t) : true;
          const isCached = withCache && p < t;
          const x = 20 + p * size;
          const y = 20 + t * size;
          const color = isCurrent ? '#2563eb' : (isCached ? '#86efac' : '#f3f4f6');
          html += `<rect x="${x}" y="${y}" width="${size - 2}" height="${size - 2}" fill="${color}" stroke="#d1d5db"/>`;
          count += isCurrent ? 1 : 0;
        }
      }
      document.getElementById(elId).innerHTML = html;
      return count;
    }
    // At step s, without-cache recomputes 0..s positions
    // with-cache only computes position s
    let totalNo = 0, totalYes = 0;
    for (let t = 0; t <= step; t++) {
      totalNo += t + 1;
      totalYes += 1;
    }
    draw('kv-no', false);
    draw('kv-yes', true);
    container.querySelector('#kv-out').textContent =
      `Step ${step}/${N - 1}. Total K/V computations: without cache = ${totalNo}, with cache = ${totalYes}. ` +
      `Ratio: ${(totalNo / totalYes).toFixed(1)}×`;
  }
  async function play() {
    if (playing) return;
    playing = true;
    const N = +container.querySelector('#kv-n').value;
    for (step = 0; step < N; step++) {
      render();
      await new Promise(r => setTimeout(r, 500));
    }
    playing = false;
  }
  container.querySelector('#kv-n').addEventListener('input', () => { step = 0; render(); });
  container.querySelector('#kv-play').addEventListener('click', play);
  container.querySelector('#kv-reset').addEventListener('click', () => { step = 0; render(); });
  render();
});
