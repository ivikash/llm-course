// viz/causal_mask.js
// Small interactive showing the causal mask pattern:
// user picks sequence length, sees which positions each row can attend to.

registerViz('causal_mask', function (container) {
  container.innerHTML = `
    <p class="viz-title">Causal (autoregressive) mask</p>
    <p class="viz-sub">In GPT-style models, each token can only attend to past tokens and itself. Slide the length; hover cells to see which.</p>
    <div class="viz-controls">
      <label>Sequence length T: <input id="cm-t" type="range" min="2" max="16" value="8"><span id="cm-t-out">8</span></label>
    </div>
    <svg id="cm-svg" viewBox="0 0 440 440" width="440" height="440" style="background:#fff;border:1px solid #e5e7eb;border-radius:4px;max-width:100%"></svg>
    <p class="viz-readout" id="cm-out">Hover a cell to inspect.</p>
  `;
  const svg = container.querySelector('#cm-svg');
  const tEl = container.querySelector('#cm-t');
  const tOut = container.querySelector('#cm-t-out');
  const out = container.querySelector('#cm-out');

  function render() {
    const T = +tEl.value;
    tOut.textContent = T;
    const cell = 400 / T;
    let html = '';
    for (let i = 0; i < T; i++) {
      for (let j = 0; j < T; j++) {
        const allowed = j <= i;
        const x = 30 + j * cell, y = 30 + i * cell;
        const color = allowed ? '#bbf7d0' : '#fecaca';
        const text = allowed ? '✓' : '−∞';
        html += `<g data-i="${i}" data-j="${j}">`;
        html += `<rect x="${x}" y="${y}" width="${cell}" height="${cell}" fill="${color}" stroke="#fff"/>`;
        if (cell > 20) html += `<text x="${x + cell/2}" y="${y + cell/2 + 3}" font-size="${Math.min(14, cell/2)}" text-anchor="middle" fill="#111">${text}</text>`;
        html += `</g>`;
      }
    }
    // axes
    html += `<text x="220" y="20" font-size="11" text-anchor="middle" fill="#374151">keys (attended-to positions) →</text>`;
    html += `<text x="14" y="230" font-size="11" text-anchor="middle" fill="#374151" transform="rotate(-90 14 230)">queries (attending positions)</text>`;
    for (let i = 0; i < T; i++) {
      html += `<text x="${30 + i * cell + cell/2}" y="424" font-size="10" text-anchor="middle" fill="#6b7280">${i}</text>`;
      html += `<text x="24" y="${30 + i * cell + cell/2 + 3}" font-size="10" text-anchor="end" fill="#6b7280">${i}</text>`;
    }
    svg.innerHTML = html;

    svg.querySelectorAll('g[data-i]').forEach(g => {
      g.addEventListener('mouseenter', () => {
        const i = +g.dataset.i, j = +g.dataset.j;
        const allowed = j <= i;
        out.textContent = `query pos ${i} → key pos ${j}: ${allowed ? 'ALLOWED' : 'BLOCKED (would leak future)'}`;
      });
    });
  }
  tEl.addEventListener('input', render);
  render();
});
