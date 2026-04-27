// viz/lr_schedule.js
// Interactive warmup + cosine LR schedule.

registerViz('lr_schedule', function (container) {
  container.innerHTML = `
    <p class="viz-title">Learning rate schedule: warmup + cosine decay</p>
    <p class="viz-sub">The standard LR curve used in GPT-2, Llama, and nanoGPT. Adjust the knobs. The line is your schedule.</p>
    <div class="viz-controls">
      <label>max LR: <input id="ls-mlr" type="range" min="1e-5" max="1e-2" step="1e-5" value="6e-4"><span id="ls-mlr-out">6.0e-4</span></label>
      <label>min LR: <input id="ls-nlr" type="range" min="1e-6" max="1e-3" step="1e-6" value="6e-5"><span id="ls-nlr-out">6.0e-5</span></label>
      <label>warmup steps: <input id="ls-wu" type="range" min="0" max="10000" step="100" value="2000"><span id="ls-wu-out">2000</span></label>
      <label>total steps: <input id="ls-ts" type="range" min="10000" max="1000000" step="10000" value="600000"><span id="ls-ts-out">600000</span></label>
    </div>
    <svg id="ls-svg" viewBox="0 0 600 300" width="600" height="300" style="background:#fff;border:1px solid #e5e7eb;border-radius:4px;max-width:100%"></svg>
    <p class="viz-readout" id="ls-out"></p>
  `;
  function getLr(it, maxLr, minLr, warmup, total) {
    if (it < warmup) return maxLr * (it + 1) / (warmup + 1);
    if (it > total) return minLr;
    const decayRatio = (it - warmup) / (total - warmup);
    const coeff = 0.5 * (1.0 + Math.cos(Math.PI * decayRatio));
    return minLr + coeff * (maxLr - minLr);
  }
  function render() {
    const mlr = parseFloat(container.querySelector('#ls-mlr').value);
    const nlr = parseFloat(container.querySelector('#ls-nlr').value);
    const wu = parseFloat(container.querySelector('#ls-wu').value);
    const ts = parseFloat(container.querySelector('#ls-ts').value);
    container.querySelector('#ls-mlr-out').textContent = mlr.toExponential(1);
    container.querySelector('#ls-nlr-out').textContent = nlr.toExponential(1);
    container.querySelector('#ls-wu-out').textContent = wu.toLocaleString();
    container.querySelector('#ls-ts-out').textContent = ts.toLocaleString();

    const points = 200;
    let path = '';
    for (let i = 0; i < points; i++) {
      const it = (i / (points - 1)) * ts;
      const lr = getLr(it, mlr, nlr, wu, ts);
      const x = 40 + (it / ts) * 540;
      const y = 270 - (lr / mlr) * 240;
      path += (path ? ' L ' : 'M ') + x.toFixed(1) + ' ' + y.toFixed(1);
    }
    const warmupX = 40 + (wu / ts) * 540;
    container.querySelector('#ls-svg').innerHTML = `
      <line x1="40" y1="270" x2="580" y2="270" stroke="#6b7280"/>
      <line x1="40" y1="30" x2="40" y2="270" stroke="#6b7280"/>
      <line x1="${warmupX}" y1="30" x2="${warmupX}" y2="270" stroke="#10b981" stroke-dasharray="3,3" opacity="0.5"/>
      <text x="${warmupX + 4}" y="20" font-size="11" fill="#10b981">warmup ends</text>
      <text x="300" y="294" font-size="11" fill="#6b7280" text-anchor="middle">training step</text>
      <text x="10" y="150" font-size="11" fill="#6b7280" transform="rotate(-90 10 150)">LR</text>
      <text x="32" y="36" font-size="10" fill="#6b7280" text-anchor="end">${mlr.toExponential(1)}</text>
      <text x="32" y="274" font-size="10" fill="#6b7280" text-anchor="end">${nlr.toExponential(1)}</text>
      <path d="${path}" fill="none" stroke="#2563eb" stroke-width="2.5"/>
    `;
    container.querySelector('#ls-out').textContent =
      `Formula: linear from 0 → ${mlr.toExponential(2)} over ${wu.toLocaleString()} steps, ` +
      `then cosine decay to ${nlr.toExponential(2)} over the remaining ${(ts - wu).toLocaleString()} steps.`;
  }
  container.querySelectorAll('input').forEach(el => el.addEventListener('input', render));
  render();
});
