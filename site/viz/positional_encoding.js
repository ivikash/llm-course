// viz/positional_encoding.js
// Sinusoidal positional encoding visualizer.
// Shows the classic stripes-at-different-frequencies picture.

registerViz('positional_encoding', function (container) {
  container.innerHTML = `
    <p class="viz-title">Sinusoidal positional encoding</p>
    <p class="viz-sub">Each row = one token's positional vector. Each column = one embedding dimension. Fast-oscillating dims (left) encode fine-grained position; slow dims (right) encode coarse.</p>
    <div class="viz-controls">
      <label>Positions (T): <input id="pe-t" type="range" min="8" max="128" value="64"><span id="pe-t-out">64</span></label>
      <label>Dims (d): <input id="pe-d" type="range" min="8" max="128" value="64"><span id="pe-d-out">64</span></label>
    </div>
    <canvas id="pe-canvas" width="512" height="256" style="image-rendering:pixelated;border:1px solid #e5e7eb;border-radius:4px;max-width:100%"></canvas>
    <p class="viz-readout" id="pe-out">Positive values = dark blue, negative = dark red. The pattern is deterministic — no training needed.</p>
  `;
  const cvs = container.querySelector('#pe-canvas');
  const ctx = cvs.getContext('2d');
  const tEl = container.querySelector('#pe-t');
  const dEl = container.querySelector('#pe-d');

  function render() {
    const T = +tEl.value, d = +dEl.value;
    container.querySelector('#pe-t-out').textContent = T;
    container.querySelector('#pe-d-out').textContent = d;
    const pixelW = Math.max(2, Math.floor(cvs.width / d));
    const pixelH = Math.max(2, Math.floor(cvs.height / T));
    cvs.width = pixelW * d;
    cvs.height = pixelH * T;
    for (let pos = 0; pos < T; pos++) {
      for (let i = 0; i < d; i++) {
        const isEven = i % 2 === 0;
        const freqIdx = Math.floor(i / 2);
        const angle = pos / Math.pow(10000, (2 * freqIdx) / d);
        const val = isEven ? Math.sin(angle) : Math.cos(angle);
        const t = (val + 1) / 2;
        const r = Math.floor(255 * (1 - t) * 0.9 + 128 * t);
        const g = Math.floor(200 * Math.abs(0.5 - t) * 2);
        const b = Math.floor(255 * t * 0.9 + 128 * (1 - t));
        ctx.fillStyle = `rgb(${r},${g},${b})`;
        ctx.fillRect(i * pixelW, pos * pixelH, pixelW, pixelH);
      }
    }
  }
  tEl.addEventListener('input', render);
  dEl.addEventListener('input', render);
  render();
});
