// viz/softmax_temperature.js
// Shows softmax of a fixed set of logits as a bar chart.
// Slider: temperature. Reveals how temperature flattens/sharpens.

registerViz('softmax_temperature', function (container) {
  container.innerHTML = `
    <p class="viz-title">Softmax temperature</p>
    <p class="viz-sub">The LLM "temperature" slider in one picture. Low T → confident (peaky). High T → random (flat).</p>
    <div class="viz-controls">
      <label>Temperature: <input id="sm-t" type="range" min="0.05" max="5" step="0.05" value="1"><span id="sm-t-out">1.00</span></label>
      <label>Logits (edit): <input id="sm-logits" type="text" value="1.0, 2.0, 3.0, 1.5, 0.5, 2.5" style="width:220px"></label>
    </div>
    <svg id="sm-svg" viewBox="0 0 600 260" width="600" height="260" style="background:#fff;border:1px solid #e5e7eb;border-radius:4px;max-width:100%"></svg>
    <p class="viz-readout" id="sm-out"></p>
  `;
  const tEl = container.querySelector('#sm-t');
  const tOut = container.querySelector('#sm-t-out');
  const logitsEl = container.querySelector('#sm-logits');
  const svg = container.querySelector('#sm-svg');
  const out = container.querySelector('#sm-out');

  function softmax(logits, T) {
    const scaled = logits.map(l => l / T);
    const maxL = Math.max(...scaled);
    const exps = scaled.map(l => Math.exp(l - maxL));
    const sum = exps.reduce((a, b) => a + b);
    return exps.map(e => e / sum);
  }

  function render() {
    const T = parseFloat(tEl.value);
    tOut.textContent = T.toFixed(2);
    const logits = logitsEl.value.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
    if (logits.length < 2) return;
    const probs = softmax(logits, T);
    const n = logits.length;
    const barW = 520 / n;
    const maxH = 200;
    const bars = probs.map((p, i) => {
      const h = p * maxH;
      const x = 40 + i * barW;
      const y = 220 - h;
      const color = p > 0.5 ? '#059669' : p > 0.2 ? '#2563eb' : '#9ca3af';
      return `
        <rect x="${x + 4}" y="${y}" width="${barW - 8}" height="${h}" fill="${color}"/>
        <text x="${x + barW / 2}" y="${y - 4}" font-size="11" text-anchor="middle" fill="#111">${(p * 100).toFixed(1)}%</text>
        <text x="${x + barW / 2}" y="238" font-size="10" text-anchor="middle" fill="#6b7280">logit=${logits[i].toFixed(1)}</text>
      `;
    }).join('');
    svg.innerHTML = `
      <line x1="40" y1="220" x2="560" y2="220" stroke="#6b7280"/>
      <text x="8" y="24" font-size="11" fill="#6b7280">probability</text>
      ${bars}
    `;
    // entropy as a gauge of "sharpness"
    const H = -probs.reduce((s, p) => s + (p > 0 ? p * Math.log(p) : 0), 0);
    const Hmax = Math.log(n);
    out.textContent = `entropy = ${H.toFixed(3)} / ${Hmax.toFixed(3)} (uniform)   ` +
      `→ ${(H / Hmax * 100).toFixed(0)}% as uncertain as uniform random`;
  }

  tEl.addEventListener('input', render);
  logitsEl.addEventListener('input', render);
  render();
});
