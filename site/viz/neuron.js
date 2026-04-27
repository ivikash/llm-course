// viz/neuron.js
// Single-neuron playground. Sliders for weights + bias, pick activation.
// Shows output as a function of input x.

registerViz('neuron', function (container) {
  container.innerHTML = `
    <p class="viz-title">A single neuron</p>
    <p class="viz-sub">y = activation(w · x + b). Play with weights and bias. Change the activation. See the output curve.</p>
    <div class="viz-controls">
      <label>w: <input id="n-w" type="range" min="-3" max="3" step="0.1" value="1"><span id="n-w-out">1.0</span></label>
      <label>b: <input id="n-b" type="range" min="-3" max="3" step="0.1" value="0"><span id="n-b-out">0.0</span></label>
      <label>activation:
        <select id="n-act">
          <option value="identity">identity (linear)</option>
          <option value="relu">ReLU</option>
          <option value="gelu">GELU</option>
          <option value="silu">SiLU / Swish</option>
          <option value="tanh">tanh</option>
          <option value="sigmoid">sigmoid</option>
        </select>
      </label>
    </div>
    <svg id="n-svg" viewBox="0 0 600 280" width="600" height="280" style="background:#fff;border:1px solid #e5e7eb;border-radius:4px;max-width:100%"></svg>
    <p class="viz-readout" id="n-out"></p>
  `;
  const svg = container.querySelector('#n-svg');
  const wEl = container.querySelector('#n-w');
  const bEl = container.querySelector('#n-b');
  const actEl = container.querySelector('#n-act');
  const out = container.querySelector('#n-out');
  const wOut = container.querySelector('#n-w-out');
  const bOut = container.querySelector('#n-b-out');

  const activations = {
    identity: x => x,
    relu: x => Math.max(0, x),
    gelu: x => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x * x * x))),
    silu: x => x / (1 + Math.exp(-x)),
    tanh: x => Math.tanh(x),
    sigmoid: x => 1 / (1 + Math.exp(-x)),
  };

  function xToPx(x) { return 30 + (x + 4) * (540 / 8); }  // x in [-4,4] → [30,570]
  function yToPx(y) { return 140 - y * 30; }              // y visual scale

  function render() {
    const w = parseFloat(wEl.value), b = parseFloat(bEl.value);
    const act = actEl.value;
    wOut.textContent = w.toFixed(1);
    bOut.textContent = b.toFixed(1);
    const f = activations[act];
    let path = '';
    for (let x = -4; x <= 4; x += 0.05) {
      const y = f(w * x + b);
      path += (path ? ' L ' : 'M ') + xToPx(x).toFixed(1) + ' ' + yToPx(y).toFixed(1);
    }
    // axes
    svg.innerHTML = `
      <line x1="30" y1="140" x2="570" y2="140" stroke="#d1d5db"/>
      <line x1="${xToPx(0)}" y1="10" x2="${xToPx(0)}" y2="270" stroke="#d1d5db"/>
      <text x="560" y="132" font-size="10" fill="#6b7280">x</text>
      <text x="${xToPx(0) + 4}" y="18" font-size="10" fill="#6b7280">y</text>
      ${[-4,-2,2,4].map(v => `<text x="${xToPx(v)}" y="156" font-size="10" fill="#6b7280" text-anchor="middle">${v}</text>`).join('')}
      ${[-2,2].map(v => `<text x="${xToPx(0)-12}" y="${yToPx(v)+3}" font-size="10" fill="#6b7280" text-anchor="end">${v}</text>`).join('')}
      <path d="${path}" fill="none" stroke="#2563eb" stroke-width="2.5"/>
    `;
    // readout with sample input
    const sample = 2;
    const pre = w * sample + b;
    const post = f(pre);
    out.textContent = `Example: with x=${sample}, w·x+b = ${pre.toFixed(2)}, activation(${pre.toFixed(2)}) = ${post.toFixed(3)}`;
  }

  [wEl, bEl, actEl].forEach(el => el.addEventListener('input', render));
  render();
});
