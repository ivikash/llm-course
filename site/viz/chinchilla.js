// viz/chinchilla.js
// Interactive Chinchilla scaling: for budget C FLOPs, optimal N (params) and D (tokens).

registerViz('chinchilla', function (container) {
  container.innerHTML = `
    <p class="viz-title">Chinchilla-optimal model size</p>
    <p class="viz-sub">Given a compute budget C (FLOPs), what's the best split between model size N and data D? Chinchilla says: 20 tokens per parameter, roughly.</p>
    <div class="viz-controls">
      <label>Compute budget (FLOPs):
        <select id="ch-c">
          <option value="1e20">1e20 (small research run)</option>
          <option value="1e22" selected>1e22 (GPT-2 scale)</option>
          <option value="3e23">3e23 (GPT-3)</option>
          <option value="5e24">5e24 (Chinchilla 70B)</option>
          <option value="2e25">2e25 (GPT-4 est.)</option>
          <option value="1e26">1e26 (GPT-5 est.)</option>
        </select>
      </label>
    </div>
    <svg id="ch-svg" viewBox="0 0 740 320" width="740" height="320" style="background:#fff;border:1px solid #e5e7eb;border-radius:4px;max-width:100%;margin-top:12px"></svg>
    <p class="viz-readout" id="ch-out"></p>
  `;

  function render() {
    const C = parseFloat(container.querySelector('#ch-c').value);
    // Chinchilla: C = 6 * N * D (flops), optimal is D ≈ 20N
    // So N_opt = sqrt(C / 120), D_opt = 20 * N_opt
    const N_opt = Math.sqrt(C / 120);
    const D_opt = 20 * N_opt;

    const svg = container.querySelector('#ch-svg');
    const W = 740, H = 320;
    const padL = 70, padR = 40, padT = 40, padB = 50;
    const plotW = W - padL - padR, plotH = H - padT - padB;

    // Plot loss as function of N/D ratio for fixed C
    // Loss ~ A/N^α + B/D^β + L_∞, Chinchilla α≈0.34, β≈0.28
    // Use simplified: loss = 406.4/N^0.34 + 410.7/D^0.28 + 1.69 (Chinchilla paper)
    const points = [];
    // Sweep N from 1e7 to 1e11
    const Nmin = Math.max(1e7, Math.sqrt(C / 120) / 100);
    const Nmax = Math.min(1e12, Math.sqrt(C / 120) * 100);
    const steps = 100;
    for (let i = 0; i < steps; i++) {
      const logN = Math.log10(Nmin) + (Math.log10(Nmax) - Math.log10(Nmin)) * i / (steps - 1);
      const N = Math.pow(10, logN);
      const D = C / (6 * N);
      const loss = 406.4 / Math.pow(N, 0.34) + 410.7 / Math.pow(D, 0.28) + 1.69;
      points.push({ N, D, loss });
    }
    const lossMin = Math.min(...points.map(p => p.loss));
    const lossMax = Math.max(...points.map(p => p.loss));
    const logNmin = Math.log10(Nmin), logNmax = Math.log10(Nmax);

    let html = '';
    // axes
    html += `<line x1="${padL}" y1="${padT + plotH}" x2="${W - padR}" y2="${padT + plotH}" stroke="#6b7280"/>`;
    html += `<line x1="${padL}" y1="${padT}" x2="${padL}" y2="${padT + plotH}" stroke="#6b7280"/>`;
    // x ticks
    for (let i = 0; i <= Math.ceil(logNmax - logNmin); i++) {
      const logN = logNmin + i;
      const x = padL + (logN - logNmin) / (logNmax - logNmin) * plotW;
      html += `<line x1="${x}" y1="${padT + plotH}" x2="${x}" y2="${padT + plotH + 5}" stroke="#6b7280"/>`;
      html += `<text x="${x}" y="${padT + plotH + 18}" font-size="10" text-anchor="middle" fill="#6b7280">10^${logN.toFixed(0)}</text>`;
    }
    html += `<text x="${W/2}" y="${H - 10}" font-size="11" text-anchor="middle" fill="#374151">N (parameters)</text>`;
    html += `<text x="${padL - 45}" y="${padT + plotH/2}" font-size="11" fill="#374151" transform="rotate(-90 ${padL - 45} ${padT + plotH/2})">loss</text>`;

    // curve
    let pathD = '';
    points.forEach((p, i) => {
      const x = padL + (Math.log10(p.N) - logNmin) / (logNmax - logNmin) * plotW;
      const y = padT + (1 - (lossMax - p.loss) / (lossMax - lossMin + 0.01)) * plotH;
      pathD += (i === 0 ? 'M' : 'L') + x + ' ' + y + ' ';
    });
    html += `<path d="${pathD}" stroke="#2563eb" stroke-width="2" fill="none"/>`;

    // optimal point
    const x_opt = padL + (Math.log10(N_opt) - logNmin) / (logNmax - logNmin) * plotW;
    const y_opt = padT + (1 - (lossMax - lossMin) / (lossMax - lossMin + 0.01)) * plotH;
    html += `<circle cx="${x_opt}" cy="${y_opt}" r="6" fill="#dc2626" stroke="#fff" stroke-width="2"/>`;
    html += `<text x="${x_opt}" y="${y_opt - 12}" font-size="11" text-anchor="middle" fill="#dc2626" font-weight="bold">Chinchilla-optimal</text>`;

    // GPT-3 marker (too few tokens — right of optimum on this plot)
    svg.innerHTML = html;

    const param_b = N_opt / 1e9;
    const tokens_b = D_opt / 1e9;
    container.querySelector('#ch-out').innerHTML =
      `For compute budget <b>${C.toExponential(0)} FLOPs</b>: ` +
      `optimal <b>N = ${param_b < 1 ? (param_b*1000).toFixed(0) + 'M' : param_b.toFixed(1) + 'B'}</b> params, ` +
      `<b>D = ${tokens_b < 1 ? (tokens_b*1000).toFixed(0) + 'M' : tokens_b.toFixed(1) + 'B'}</b> tokens. ` +
      `Everyone pre-Chinchilla (2022) trained models too big for their data budget. Example: GPT-3 (175B, 300B tokens) should have been ~70B, ~1.4T tokens.`;
  }
  container.querySelector('#ch-c').addEventListener('change', render);
  render();
});
