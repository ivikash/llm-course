// viz/training_curve_simulator.js
// Simulated training curve. User tweaks LR, batch size, and model size;
// sees loss curve shape change, with annotations for healthy/unhealthy patterns.

registerViz('training_curve_simulator', function (container) {
  container.innerHTML = `
    <p class="viz-title">Training curve simulator</p>
    <p class="viz-sub">Watch how hyperparameters shape your loss curve. Find the patterns you'll see in wandb.</p>
    <div class="viz-controls">
      <label>LR factor: <input id="tcs-lr" type="range" min="0.1" max="5" step="0.1" value="1"><span id="tcs-lr-out">1.0</span></label>
      <label>Model size: <input id="tcs-ms" type="range" min="1" max="10" step="1" value="3"><span id="tcs-ms-out">3</span></label>
      <label>Noise (small batch): <input id="tcs-n" type="range" min="0" max="1" step="0.05" value="0.2"><span id="tcs-n-out">0.20</span></label>
      <label>Seeds: <input id="tcs-s" type="range" min="1" max="5" step="1" value="3"><span id="tcs-s-out">3</span></label>
    </div>
    <svg id="tcs-svg" viewBox="0 0 600 320" width="600" height="320" style="background:#fff;border:1px solid #e5e7eb;border-radius:4px;max-width:100%"></svg>
    <p class="viz-readout" id="tcs-out"></p>
  `;
  function loss(step, lr, ms, noise, seed) {
    const warm = 200;
    if (step < warm) return 10 - 5 * step / warm;
    const asymp = 2 + 1.5 / ms;
    // base: exponential decay to asymp
    let l = asymp + (10 - asymp) * Math.exp(-(step - warm) * 0.002 * Math.sqrt(ms));
    // lr too high → diverge
    if (lr > 3.5) return step > warm + 200 ? 50 * Math.min(1, (step - warm - 200) / 1000) + l : l;
    if (lr > 2.5) l *= 1 + (lr - 2.5) * 0.5 * Math.sin(step * 0.01 + seed);
    // lr too low → slow
    if (lr < 0.3) l = 10 - (10 - l) * lr / 0.3;
    // seed-dependent noise
    const r = Math.sin(step * 0.3 + seed * 100) + Math.cos(step * 0.11 + seed * 50);
    return l + noise * 0.3 * r;
  }
  function render() {
    const lr = parseFloat(container.querySelector('#tcs-lr').value);
    const ms = parseInt(container.querySelector('#tcs-ms').value);
    const noise = parseFloat(container.querySelector('#tcs-n').value);
    const seeds = parseInt(container.querySelector('#tcs-s').value);
    container.querySelector('#tcs-lr-out').textContent = lr.toFixed(1);
    container.querySelector('#tcs-ms-out').textContent = ms;
    container.querySelector('#tcs-n-out').textContent = noise.toFixed(2);
    container.querySelector('#tcs-s-out').textContent = seeds;

    const numSteps = 2000;
    let paths = '';
    const colors = ['#2563eb','#dc2626','#059669','#d97706','#7c3aed'];
    let finalLosses = [];
    for (let s = 0; s < seeds; s++) {
      let path = '';
      let sLoss = 0;
      for (let step = 0; step < numSteps; step += 10) {
        const l = loss(step, lr, ms, noise, s);
        if (!isFinite(l) || l > 60) continue;
        sLoss = l;
        const x = 40 + (step / numSteps) * 540;
        const y = 280 - Math.min(Math.max(l, 0), 12) * 22;
        path += (path ? ' L ' : 'M ') + x.toFixed(1) + ' ' + y.toFixed(1);
      }
      finalLosses.push(sLoss);
      paths += `<path d="${path}" fill="none" stroke="${colors[s % colors.length]}" stroke-width="1.5" opacity="0.8"/>`;
    }
    let diag = '';
    const avgFinal = finalLosses.reduce((a, b) => a + b, 0) / seeds;
    if (lr > 3.5) diag = '⚠️ DIVERGED — loss blowing up. LR too high.';
    else if (lr > 2.5) diag = '⚠️ Unstable oscillations. Consider LR warmup or lowering.';
    else if (lr < 0.3) diag = 'Too slow (zzz). LR is low; loss hasn\u2019t plateaued yet.';
    else if (avgFinal < 3) diag = '✅ Healthy: smooth decay, nice plateau.';
    else diag = '✅ Training healthily, more steps would help.';

    container.querySelector('#tcs-svg').innerHTML = `
      ${[2,4,6,8,10].map(y => `<line x1="40" y1="${280-y*22}" x2="580" y2="${280-y*22}" stroke="#f3f4f6"/><text x="32" y="${285-y*22}" font-size="10" fill="#6b7280" text-anchor="end">${y}</text>`).join('')}
      <line x1="40" y1="280" x2="580" y2="280" stroke="#6b7280"/>
      <text x="300" y="306" font-size="11" fill="#6b7280" text-anchor="middle">training step</text>
      <text x="12" y="150" font-size="11" fill="#6b7280" transform="rotate(-90 12 150)">loss</text>
      ${paths}
    `;
    container.querySelector('#tcs-out').textContent = `${diag}   final loss ≈ ${avgFinal.toFixed(2)}, seed variance ≈ ${finalLosses.length > 1 ? (Math.max(...finalLosses)-Math.min(...finalLosses)).toFixed(2) : '-'}`;
  }
  container.querySelectorAll('input').forEach(el => el.addEventListener('input', render));
  render();
});
