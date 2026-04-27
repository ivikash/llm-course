// viz/dropout_viz.js
// Dropout: random neurons zeroed per forward pass. Watch it happen.

registerViz('dropout_viz', function (container) {
  container.innerHTML = `
    <p class="viz-title">Dropout in action</p>
    <p class="viz-sub">Each forward pass, a random fraction of neurons is zeroed. Forces the network to be robust — no single neuron can be relied on.</p>
    <div class="viz-controls">
      <label>Dropout p: <input id="dv-p" type="range" min="0" max="0.9" step="0.05" value="0.3"><span id="dv-p-out">0.30</span></label>
      <label>Mode:
        <select id="dv-m">
          <option value="train" selected>Training (dropout ON)</option>
          <option value="eval">Eval (dropout OFF)</option>
        </select>
      </label>
      <button id="dv-step">New forward pass</button>
    </div>
    <svg id="dv-svg" viewBox="0 0 600 320" width="600" height="320" style="background:#fff;border:1px solid #e5e7eb;border-radius:4px;max-width:100%"></svg>
    <p class="viz-readout" id="dv-out"></p>
  `;
  const layers = [4, 8, 8, 3];

  function render() {
    const p = +container.querySelector('#dv-p').value;
    const mode = container.querySelector('#dv-m').value;
    container.querySelector('#dv-p-out').textContent = p.toFixed(2);
    const svg = container.querySelector('#dv-svg');
    const W = 600, H = 320;
    const gap = W / (layers.length + 1);
    const positions = [];
    for (let l = 0; l < layers.length; l++) {
      const n = layers[l];
      const yStep = H / (n + 1);
      const pos = [];
      for (let i = 0; i < n; i++) {
        pos.push({ x: gap * (l + 1), y: yStep * (i + 1) });
      }
      positions.push(pos);
    }
    // dropout mask (skip input and output layers)
    const masks = positions.map((layer, l) =>
      (l === 0 || l === positions.length - 1 || mode === 'eval')
        ? layer.map(() => 1)
        : layer.map(() => Math.random() > p ? 1 : 0)
    );
    // draw edges
    let edges = '';
    for (let l = 0; l < positions.length - 1; l++) {
      for (let i = 0; i < positions[l].length; i++) {
        for (let j = 0; j < positions[l + 1].length; j++) {
          const alive = masks[l][i] && masks[l + 1][j];
          edges += `<line x1="${positions[l][i].x}" y1="${positions[l][i].y}" x2="${positions[l+1][j].x}" y2="${positions[l+1][j].y}" stroke="${alive ? '#2563eb' : '#e5e7eb'}" stroke-width="${alive ? 0.8 : 0.3}"/>`;
        }
      }
    }
    // draw neurons
    let neurons = '';
    positions.forEach((layer, l) => {
      layer.forEach((p, i) => {
        const alive = masks[l][i];
        const color = alive ? '#10b981' : '#ef4444';
        const r = alive ? 8 : 6;
        neurons += `<circle cx="${p.x}" cy="${p.y}" r="${r}" fill="${color}"/>`;
        if (!alive) {
          // draw an X through
          neurons += `<line x1="${p.x-5}" y1="${p.y-5}" x2="${p.x+5}" y2="${p.y+5}" stroke="#fff" stroke-width="2"/>`;
          neurons += `<line x1="${p.x-5}" y1="${p.y+5}" x2="${p.x+5}" y2="${p.y-5}" stroke="#fff" stroke-width="2"/>`;
        }
      });
    });
    svg.innerHTML = edges + neurons;
    const totalNeurons = masks.flat().length;
    const aliveCount = masks.flat().filter(v => v).length;
    const dropped = totalNeurons - aliveCount;
    container.querySelector('#dv-out').textContent = `${aliveCount}/${totalNeurons} neurons active, ${dropped} dropped this pass. ` +
      (mode === 'eval' ? 'In eval mode, ALL neurons active (no dropout). Outputs scaled by (1-p) at train time instead.' :
       `Different neurons drop each pass → the network can't rely on any single one → better generalization.`);
  }
  container.querySelector('#dv-p').addEventListener('input', render);
  container.querySelector('#dv-m').addEventListener('change', render);
  container.querySelector('#dv-step').addEventListener('click', render);
  render();
});
