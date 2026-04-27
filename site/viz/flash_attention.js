// viz/flash_attention.js
// Visualize flash attention's tile-based computation vs naive O(T²) memory.

registerViz('flash_attention', function (container) {
  container.innerHTML = `
    <p class="viz-title">Flash Attention: same math, tiles in SRAM</p>
    <p class="viz-sub">Naive attention materializes the full T×T scores matrix in slow HBM memory. Flash Attention processes it in small tiles in fast on-chip SRAM.</p>
    <div class="viz-controls">
      <label>Sequence T: <input id="fa-t" type="range" min="8" max="64" step="4" value="32"><span id="fa-t-out">32</span></label>
      <label>Tile size: <input id="fa-b" type="range" min="4" max="16" step="2" value="8"><span id="fa-b-out">8</span></label>
      <button id="fa-play">▶ Play Flash Attention</button>
    </div>
    <div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center">
      <div>
        <div style="text-align:center;font-weight:bold;color:#dc2626;margin-bottom:4px">Naive: full T×T in HBM</div>
        <canvas id="fa-naive" width="260" height="260" style="image-rendering:pixelated;border:1px solid #d1d5db;border-radius:4px"></canvas>
      </div>
      <div>
        <div style="text-align:center;font-weight:bold;color:#10b981;margin-bottom:4px">Flash: tile-by-tile in SRAM</div>
        <canvas id="fa-flash" width="260" height="260" style="image-rendering:pixelated;border:1px solid #d1d5db;border-radius:4px"></canvas>
      </div>
    </div>
    <p class="viz-readout" id="fa-out"></p>
  `;
  const naive = container.querySelector('#fa-naive').getContext('2d');
  const flash = container.querySelector('#fa-flash').getContext('2d');
  let currentTile = null;

  function draw() {
    const T = +container.querySelector('#fa-t').value;
    const B = +container.querySelector('#fa-b').value;
    container.querySelector('#fa-t-out').textContent = T;
    container.querySelector('#fa-b-out').textContent = B;
    const cell = 260 / T;

    // Naive: full matrix always materialized
    naive.clearRect(0, 0, 260, 260);
    for (let i = 0; i < T; i++) {
      for (let j = 0; j < T; j++) {
        naive.fillStyle = j > i ? '#f3f4f6' : '#dc2626';
        naive.fillRect(j * cell, i * cell, cell - 0.5, cell - 0.5);
      }
    }

    // Flash: only the current tile
    flash.clearRect(0, 0, 260, 260);
    flash.fillStyle = '#f3f4f6';
    flash.fillRect(0, 0, 260, 260);
    if (currentTile) {
      const { qTile, kvTile } = currentTile;
      const x = kvTile * B * cell;
      const y = qTile * B * cell;
      const w = Math.min(B, T - kvTile * B) * cell;
      const h = Math.min(B, T - qTile * B) * cell;
      flash.fillStyle = '#10b981';
      flash.fillRect(x, y, w, h);
      // grid hints
      for (let i = 0; i < Math.ceil(T / B); i++) {
        flash.strokeStyle = '#d1d5db';
        flash.lineWidth = 0.5;
        flash.strokeRect(i * B * cell, 0, B * cell, T * cell);
        flash.strokeRect(0, i * B * cell, T * cell, B * cell);
      }
    } else {
      // grid only
      for (let i = 0; i < Math.ceil(T / B); i++) {
        flash.strokeStyle = '#d1d5db';
        flash.strokeRect(i * B * cell, 0, B * cell, T * cell);
        flash.strokeRect(0, i * B * cell, T * cell, B * cell);
      }
    }

    const naiveMem = T * T * 2; // bf16
    const flashMem = B * B * 2;
    container.querySelector('#fa-out').innerHTML =
      `Naive peak memory for scores: <b>${(T*T).toLocaleString()}</b> values (${(naiveMem/1024).toFixed(2)} KB bf16). ` +
      `Flash peak: <b>${B*B}</b> values (${flashMem} bytes, fits in SRAM). ` +
      `Ratio: <b>${Math.floor((T*T)/(B*B))}×</b> less memory.`;
  }

  async function play() {
    const T = +container.querySelector('#fa-t').value;
    const B = +container.querySelector('#fa-b').value;
    const tiles = Math.ceil(T / B);
    for (let q = 0; q < tiles; q++) {
      for (let k = 0; k <= q; k++) {   // causal: only attend to kv_tile <= q_tile
        currentTile = { qTile: q, kvTile: k };
        draw();
        await new Promise(r => setTimeout(r, 150));
      }
    }
    currentTile = null;
    draw();
  }

  container.querySelector('#fa-t').addEventListener('input', () => { currentTile = null; draw(); });
  container.querySelector('#fa-b').addEventListener('input', () => { currentTile = null; draw(); });
  container.querySelector('#fa-play').addEventListener('click', play);
  draw();
});
