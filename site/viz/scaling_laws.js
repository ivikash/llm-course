// viz/scaling_laws.js
// Chinchilla-style isoFLOP curves. User picks compute; shows optimal N and D.

registerViz('scaling_laws', function (container) {
  container.innerHTML = `
    <p class="viz-title">Chinchilla scaling laws</p>
    <p class="viz-sub">For each compute budget, there's an optimal (model size, data size). Slide compute; see the whole curve; see Chinchilla's prediction.</p>
    <div class="viz-controls">
      <label>Compute (FLOPs): <input id="sc-c" type="range" min="19" max="25" step="0.1" value="22"><span id="sc-c-out">1e22</span></label>
      <label>Mode: <select id="sc-m">
        <option value="chinchilla" selected>Chinchilla (≈ 20 tokens/param)</option>
        <option value="overtrain">Llama-3 style (≈ 1000 tokens/param)</option>
        <option value="gpt3">GPT-3 style (≈ 1.7 tokens/param)</option>
      </select></label>
    </div>
    <svg id="sc-svg" viewBox="0 0 600 340" width="600" height="340" style="background:#fff;border:1px solid #e5e7eb;border-radius:4px;max-width:100%"></svg>
    <p class="viz-readout" id="sc-out"></p>
  `;
  // Chinchilla formula
  function L(N, D) {
    return 1.69 + 406.4 / Math.pow(N, 0.34) + 410.7 / Math.pow(D, 0.28);
  }
  function render() {
    const logC = parseFloat(container.querySelector('#sc-c').value);
    const C = Math.pow(10, logC);
    const mode = container.querySelector('#sc-m').value;
    container.querySelector('#sc-c-out').textContent = C.toExponential(1);

    const svg = container.querySelector('#sc-svg');
    // Plot loss vs N (model size), holding compute C fixed. D = C/(6N)
    const ns = []; // log10 N
    const losses = [];
    for (let logN = 7; logN <= 12; logN += 0.03) {
      const N = Math.pow(10, logN);
      const D = C / (6 * N);
      if (D < 1e6) continue;
      const l = L(N, D);
      ns.push(logN); losses.push(l);
    }
    // Find the minimum
    const minIdx = losses.indexOf(Math.min(...losses));
    const bestLogN = ns[minIdx];
    const bestN = Math.pow(10, bestLogN);
    const bestD = C / (6 * bestN);
    const bestL = losses[minIdx];

    // Plot
    const xToPx = x => 50 + (x - 7) * (540 / 5);
    const yToPx = y => 300 - (y - 1.8) * 150;
    let path = '';
    for (let i = 0; i < ns.length; i++) {
      path += (path ? ' L ' : 'M ') + xToPx(ns[i]).toFixed(1) + ' ' + yToPx(Math.min(losses[i], 4.5)).toFixed(1);
    }
    // Mode-specific N (user's ratio):
    let userN, userRatio;
    if (mode === 'chinchilla') userRatio = 20;
    else if (mode === 'overtrain') userRatio = 1000;
    else userRatio = 1.7;
    // N such that D/N = userRatio and 6*N*D = C
    // C = 6 * N * (userRatio * N) = 6 * userRatio * N^2 → N = sqrt(C / (6 * userRatio))
    userN = Math.sqrt(C / (6 * userRatio));
    const userD = userRatio * userN;
    const userL = L(userN, userD);
    const userLogN = Math.log10(userN);

    svg.innerHTML = `
      <line x1="50" y1="300" x2="590" y2="300" stroke="#6b7280"/>
      <line x1="50" y1="20" x2="50" y2="300" stroke="#6b7280"/>
      ${[7,8,9,10,11,12].map(v => `<text x="${xToPx(v)}" y="316" font-size="10" fill="#6b7280" text-anchor="middle">10^${v}</text>`).join('')}
      ${[2,2.5,3,3.5,4].map(y => `<text x="44" y="${yToPx(y)+3}" font-size="10" fill="#6b7280" text-anchor="end">${y}</text>`).join('')}
      <text x="300" y="334" font-size="11" fill="#6b7280" text-anchor="middle">model size N (params)</text>
      <text x="22" y="160" font-size="11" fill="#6b7280" transform="rotate(-90 22 160)">loss</text>
      <path d="${path}" fill="none" stroke="#2563eb" stroke-width="2"/>
      <circle cx="${xToPx(bestLogN)}" cy="${yToPx(bestL)}" r="8" fill="#10b981"/>
      <text x="${xToPx(bestLogN)+10}" y="${yToPx(bestL)-6}" font-size="11" fill="#10b981" font-weight="bold">Chinchilla-optimal</text>
      <circle cx="${xToPx(userLogN)}" cy="${yToPx(Math.min(userL, 4.5))}" r="6" fill="#dc2626"/>
      <text x="${xToPx(userLogN)+10}" y="${yToPx(Math.min(userL, 4.5))+14}" font-size="11" fill="#dc2626" font-weight="bold">your mode</text>
    `;
    container.querySelector('#sc-out').innerHTML =
      `At ${C.toExponential(1)} FLOPs: ` +
      `<b>Chinchilla-optimal</b>: N=${(bestN/1e9).toFixed(1)}B params, D=${(bestD/1e9).toFixed(1)}B tokens, predicted loss ${bestL.toFixed(3)}. ` +
      `<b>Your mode (${userRatio} tok/param)</b>: N=${(userN/1e9).toFixed(1)}B, D=${(userD/1e9).toFixed(1)}B, loss ${userL.toFixed(3)}.`;
  }
  container.querySelectorAll('input,select').forEach(el => el.addEventListener('input', render));
  render();
});
