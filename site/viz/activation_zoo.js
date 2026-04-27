// viz/activation_zoo.js
// All common activations plotted together with their derivatives.

registerViz('activation_zoo', function (container) {
  container.innerHTML = `
    <p class="viz-title">Activation zoo</p>
    <p class="viz-sub">Each function (top) and its derivative (bottom). Notice ReLU's derivative is exactly 0 for x&lt;0 — that's the 'dying ReLU' problem. GELU and SiLU stay smooth.</p>
    <div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center">
      <svg id="az-fn" viewBox="0 0 400 240" width="400" height="240" style="background:#fff;border:1px solid #e5e7eb;border-radius:4px"></svg>
      <svg id="az-grad" viewBox="0 0 400 240" width="400" height="240" style="background:#fff;border:1px solid #e5e7eb;border-radius:4px"></svg>
    </div>
  `;

  const acts = [
    { name: 'ReLU',    color: '#ef4444', f: x => Math.max(0, x), g: x => x > 0 ? 1 : 0 },
    { name: 'GELU',    color: '#2563eb', f: x => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3))),
      g: x => { const h = 0.0001; return (0.5 * (x+h) * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * ((x+h) + 0.044715 * (x+h)**3))) -
                                 0.5 * (x-h) * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * ((x-h) + 0.044715 * (x-h)**3)))) / (2*h); } },
    { name: 'SiLU',    color: '#059669', f: x => x / (1 + Math.exp(-x)),
      g: x => { const s = 1 / (1 + Math.exp(-x)); return s + x * s * (1 - s); } },
    { name: 'tanh',    color: '#d97706', f: x => Math.tanh(x), g: x => 1 - Math.tanh(x) ** 2 },
  ];

  function drawPlot(svgEl, which) {
    const xToPx = x => 30 + (x + 4) * (360 / 8);
    const yToPx = y => 120 - y * 40;
    let html = `
      <line x1="30" y1="120" x2="390" y2="120" stroke="#d1d5db"/>
      <line x1="${xToPx(0)}" y1="10" x2="${xToPx(0)}" y2="230" stroke="#d1d5db"/>
      <text x="8" y="24" font-size="11" fill="#6b7280">${which === 'f' ? 'y' : "y'"}</text>
    `;
    let legendY = 18;
    for (const a of acts) {
      let path = '';
      for (let x = -4; x <= 4; x += 0.05) {
        const y = which === 'f' ? a.f(x) : a.g(x);
        if (!isFinite(y)) continue;
        path += (path ? ' L ' : 'M ') + xToPx(x).toFixed(1) + ' ' + yToPx(y).toFixed(1);
      }
      html += `<path d="${path}" fill="none" stroke="${a.color}" stroke-width="2"/>`;
      html += `<circle cx="345" cy="${legendY}" r="5" fill="${a.color}"/>
               <text x="355" y="${legendY + 4}" font-size="10" fill="#374151">${a.name}</text>`;
      legendY += 14;
    }
    svgEl.innerHTML = html;
  }

  drawPlot(container.querySelector('#az-fn'), 'f');
  drawPlot(container.querySelector('#az-grad'), 'g');
});
