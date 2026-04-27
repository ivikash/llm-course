// viz/cross_entropy.js
// Interactive cross-entropy loss.
// Shows: as model's probability on correct class → 1, loss → 0.
// As it → 0, loss → infinity (exponentially bad).

registerViz('cross_entropy', function (container) {
  container.innerHTML = `
    <p class="viz-title">Cross-entropy = surprise</p>
    <p class="viz-sub">The loss function of every LLM. Slide the probability the model assigned to the TRUE class. Loss = -log(p).</p>
    <div class="viz-controls">
      <label>P(correct class): <input id="ce-p" type="range" min="0.001" max="1" step="0.001" value="0.5"><span id="ce-p-out">0.50</span></label>
    </div>
    <svg id="ce-svg" viewBox="0 0 600 300" width="600" height="300" style="background:#fff;border:1px solid #e5e7eb;border-radius:4px;max-width:100%"></svg>
    <p class="viz-readout" id="ce-out"></p>
  `;
  const svg = container.querySelector('#ce-svg');
  const pEl = container.querySelector('#ce-p');
  const pOut = container.querySelector('#ce-p-out');
  const out = container.querySelector('#ce-out');

  function xToPx(p) { return 40 + p * 520; }          // p in [0,1]
  function yToPx(l) { return 260 - Math.min(l, 8) * 30; } // loss up to 8

  function render() {
    const p = parseFloat(pEl.value);
    pOut.textContent = p.toFixed(2);
    const loss = -Math.log(p);
    let path = '';
    for (let x = 0.001; x <= 1; x += 0.002) {
      const l = -Math.log(x);
      path += (path ? ' L ' : 'M ') + xToPx(x).toFixed(1) + ' ' + yToPx(l).toFixed(1);
    }
    const gridY = [0, 1, 2, 3, 4, 5, 6].map(v =>
      `<line x1="40" y1="${yToPx(v)}" x2="560" y2="${yToPx(v)}" stroke="#f3f4f6"/>` +
      `<text x="32" y="${yToPx(v) + 3}" font-size="10" fill="#6b7280" text-anchor="end">${v}</text>`).join('');
    svg.innerHTML = `
      ${gridY}
      <line x1="40" y1="260" x2="560" y2="260" stroke="#6b7280"/>
      <text x="280" y="290" font-size="11" fill="#6b7280" text-anchor="middle">p(correct class)</text>
      <text x="12" y="150" font-size="11" fill="#6b7280" transform="rotate(-90 12 150)">loss = -log(p)</text>
      ${[0.0, 0.2, 0.4, 0.6, 0.8, 1.0].map(v =>
        `<text x="${xToPx(v)}" y="275" font-size="10" fill="#6b7280" text-anchor="middle">${v.toFixed(1)}</text>`).join('')}
      <path d="${path}" fill="none" stroke="#dc2626" stroke-width="2.5"/>
      <line x1="${xToPx(p)}" y1="10" x2="${xToPx(p)}" y2="260" stroke="#059669" stroke-dasharray="4,3"/>
      <circle cx="${xToPx(p)}" cy="${yToPx(loss)}" r="6" fill="#059669"/>
      <text x="${xToPx(p)+8}" y="${yToPx(loss)-6}" font-size="11" fill="#059669" font-weight="bold">(p=${p.toFixed(2)}, loss=${loss.toFixed(2)})</text>
    `;
    const nats = loss;
    const bits = loss / Math.log(2);
    const perplexity = Math.exp(loss);
    out.textContent = `loss = ${nats.toFixed(3)} nats = ${bits.toFixed(3)} bits   ` +
                      `perplexity = ${perplexity.toFixed(2)}   ` +
                      `(model is as uncertain as choosing uniformly among ${perplexity.toFixed(1)} options)`;
  }
  pEl.addEventListener('input', render);
  render();
});
