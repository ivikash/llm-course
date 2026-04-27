// viz/sampling_strategies.js
// Interactive sampling: given a fixed logit distribution, show what
// greedy, temperature, top-k, top-p sampling each produce.

registerViz('sampling_strategies', function (container) {
  container.innerHTML = `
    <p class="viz-title">Sampling strategies compared</p>
    <p class="viz-sub">Same logits (probabilities over 12 next-token candidates). Slide each knob; see how each sampler filters the distribution.</p>
    <div class="viz-controls">
      <label>Temperature: <input id="ss-t" type="range" min="0.1" max="3" step="0.05" value="1"><span id="ss-t-out">1.00</span></label>
      <label>Top-k: <input id="ss-k" type="range" min="1" max="12" step="1" value="12"><span id="ss-k-out">12</span></label>
      <label>Top-p: <input id="ss-p" type="range" min="0.01" max="1" step="0.01" value="1"><span id="ss-p-out">1.00</span></label>
      <button id="ss-sample">Sample 20×</button>
    </div>
    <svg id="ss-svg" viewBox="0 0 600 240" width="600" height="240" style="background:#fff;border:1px solid #e5e7eb;border-radius:4px;max-width:100%"></svg>
    <p class="viz-readout" id="ss-out"></p>
  `;
  const svg = container.querySelector('#ss-svg');
  const tEl = container.querySelector('#ss-t');
  const kEl = container.querySelector('#ss-k');
  const pEl = container.querySelector('#ss-p');
  const out = container.querySelector('#ss-out');

  const tokens = ['cat', 'dog', 'fish', 'bird', 'lizard', 'car', 'tree', 'house', 'moon', 'star', 'sun', 'cloud'];
  // Hand-crafted base logits: "cat" is dominant, then animals, then objects
  const baseLogits = [4.5, 4.0, 3.5, 3.2, 2.5, 1.5, 1.3, 1.0, 0.8, 0.5, 0.3, 0.1];

  function softmax(logits, T) {
    const scaled = logits.map(l => l / T);
    const maxL = Math.max(...scaled);
    const exps = scaled.map(l => Math.exp(l - maxL));
    const sum = exps.reduce((a, b) => a + b);
    return exps.map(e => e / sum);
  }

  function applyTopK(probs, k) {
    const indexed = probs.map((p, i) => ({ p, i }));
    indexed.sort((a, b) => b.p - a.p);
    const kept = new Set(indexed.slice(0, k).map(x => x.i));
    return probs.map((p, i) => kept.has(i) ? p : 0);
  }

  function applyTopP(probs, p_threshold) {
    const indexed = probs.map((p, i) => ({ p, i }));
    indexed.sort((a, b) => b.p - a.p);
    let cum = 0;
    const kept = new Set();
    for (const x of indexed) {
      kept.add(x.i);
      cum += x.p;
      if (cum >= p_threshold) break;
    }
    return probs.map((p, i) => kept.has(i) ? p : 0);
  }

  function renormalize(probs) {
    const sum = probs.reduce((a, b) => a + b, 0) || 1;
    return probs.map(p => p / sum);
  }

  let sampleCounts = new Array(tokens.length).fill(0);

  function render() {
    const T = parseFloat(tEl.value);
    const k = parseInt(kEl.value);
    const pThresh = parseFloat(pEl.value);
    container.querySelector('#ss-t-out').textContent = T.toFixed(2);
    container.querySelector('#ss-k-out').textContent = k;
    container.querySelector('#ss-p-out').textContent = pThresh.toFixed(2);

    // Base probs
    const afterT = softmax(baseLogits, T);
    const afterK = renormalize(applyTopK(afterT, k));
    const afterP = renormalize(applyTopP(afterK, pThresh));

    // draw bars
    const n = tokens.length;
    const barW = 500 / n;
    let bars = '';
    for (let i = 0; i < n; i++) {
      const final = afterP[i];
      const h = final * 180;
      const x = 40 + i * barW;
      const kept = final > 0;
      const totalSamples = sampleCounts.reduce((a, b) => a + b, 0);
      const empFreq = totalSamples > 0 ? sampleCounts[i] / totalSamples : 0;
      bars += `<rect x="${x + 3}" y="${220 - h}" width="${barW - 6}" height="${h}" fill="${kept ? '#2563eb' : '#e5e7eb'}"/>`;
      if (empFreq > 0) {
        const eh = empFreq * 180;
        bars += `<rect x="${x + 3}" y="${220 - eh}" width="${barW - 6}" height="3" fill="#dc2626"/>`;
      }
      bars += `<text x="${x + barW/2}" y="236" font-size="10" fill="#6b7280" text-anchor="middle">${tokens[i]}</text>`;
    }
    svg.innerHTML = `
      <line x1="40" y1="220" x2="540" y2="220" stroke="#6b7280"/>
      <text x="10" y="20" font-size="11" fill="#6b7280">p</text>
      ${bars}
      <text x="540" y="14" font-size="10" fill="#2563eb" text-anchor="end">blue: model's distribution after filtering</text>
      <text x="540" y="26" font-size="10" fill="#dc2626" text-anchor="end">red: empirical frequency from Sample 20× button</text>
    `;
    const kept = afterP.filter(p => p > 0).length;
    out.textContent = `After T=${T.toFixed(2)}, top-k=${k}, top-p=${pThresh.toFixed(2)}: ${kept} tokens kept (out of ${n}).`;
  }

  function sampleOnce() {
    const T = parseFloat(tEl.value);
    const k = parseInt(kEl.value);
    const pThresh = parseFloat(pEl.value);
    const afterT = softmax(baseLogits, T);
    const afterK = renormalize(applyTopK(afterT, k));
    const afterP = renormalize(applyTopP(afterK, pThresh));
    let r = Math.random();
    for (let i = 0; i < afterP.length; i++) {
      r -= afterP[i];
      if (r <= 0) return i;
    }
    return afterP.length - 1;
  }

  tEl.addEventListener('input', render);
  kEl.addEventListener('input', render);
  pEl.addEventListener('input', render);
  container.querySelector('#ss-sample').addEventListener('click', () => {
    sampleCounts = new Array(tokens.length).fill(0);
    for (let i = 0; i < 20; i++) {
      const idx = sampleOnce();
      sampleCounts[idx]++;
    }
    render();
  });
  render();
});
