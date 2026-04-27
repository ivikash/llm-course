// viz/fsdp_shard.js
// Compare DDP vs ZeRO-1 vs ZeRO-2 vs FSDP/ZeRO-3 memory layout.

registerViz('fsdp_shard', function (container) {
  container.innerHTML = `
    <p class="viz-title">DDP vs ZeRO vs FSDP: sharding memory</p>
    <p class="viz-sub">All 4 approaches have the same math. The difference is what's duplicated on every GPU vs sharded across them.</p>
    <div class="viz-controls">
      <label>Strategy:
        <select id="fs-s">
          <option value="ddp" selected>DDP (no sharding)</option>
          <option value="z1">ZeRO-1 (shard optimizer state)</option>
          <option value="z2">ZeRO-2 (+ shard gradients)</option>
          <option value="z3">FSDP / ZeRO-3 (+ shard weights)</option>
        </select>
      </label>
      <label>GPUs: <input id="fs-n" type="range" min="2" max="16" value="4"><span id="fs-n-out">4</span></label>
      <label>Model params: <input id="fs-p" type="range" min="1" max="100" value="10"><span id="fs-p-out">10B</span></label>
    </div>
    <div id="fs-grid" style="display:flex;gap:8px;margin-top:12px;flex-wrap:wrap;justify-content:center"></div>
    <p class="viz-readout" id="fs-out"></p>
  `;
  function render() {
    const strategy = container.querySelector('#fs-s').value;
    const N = +container.querySelector('#fs-n').value;
    const P = +container.querySelector('#fs-p').value;
    container.querySelector('#fs-n-out').textContent = N;
    container.querySelector('#fs-p-out').textContent = P + 'B';

    // Per-GPU memory (GB) for bf16+AdamW:
    // weights 6 bytes/param, grads 2 bytes/param, optimizer 8 bytes/param
    const w = P * 6, g = P * 2, o = P * 8;
    const fractions = {
      ddp: { w, g, o },
      z1:  { w, g, o: o / N },
      z2:  { w, g: g / N, o: o / N },
      z3:  { w: w / N, g: g / N, o: o / N },
    }[strategy];

    const grid = container.querySelector('#fs-grid');
    grid.innerHTML = '';
    const maxMem = 160; // 160 GB — fits H100-141GB x 1
    for (let i = 0; i < N; i++) {
      const total = fractions.w + fractions.g + fractions.o;
      const wH = (fractions.w / maxMem) * 180;
      const gH = (fractions.g / maxMem) * 180;
      const oH = (fractions.o / maxMem) * 180;
      const box = document.createElement('div');
      box.style.cssText = 'display:flex;flex-direction:column;align-items:center;gap:4px';
      box.innerHTML = `
        <div style="font-size:11px;font-weight:bold">GPU ${i}</div>
        <div style="width:50px;height:180px;border:1px solid #d1d5db;border-radius:4px;background:#f3f4f6;position:relative;overflow:hidden">
          <div style="position:absolute;bottom:0;left:0;right:0;height:${wH}px;background:#3b82f6" title="weights"></div>
          <div style="position:absolute;bottom:${wH}px;left:0;right:0;height:${gH}px;background:#f59e0b" title="gradients"></div>
          <div style="position:absolute;bottom:${wH+gH}px;left:0;right:0;height:${oH}px;background:#10b981" title="optimizer state"></div>
        </div>
        <div style="font-size:10px;font-family:ui-monospace,monospace">${total.toFixed(1)} GB</div>
      `;
      grid.appendChild(box);
    }
    const totalEach = fractions.w + fractions.g + fractions.o;
    const ddpEach = w + g + o;
    const savings = ((1 - totalEach / ddpEach) * 100).toFixed(0);
    const legend = `<span style="background:#3b82f6;color:#fff;padding:1px 6px;border-radius:3px;font-size:10px">weights</span> <span style="background:#f59e0b;color:#fff;padding:1px 6px;border-radius:3px;font-size:10px">gradients</span> <span style="background:#10b981;color:#fff;padding:1px 6px;border-radius:3px;font-size:10px">optimizer</span>`;
    container.querySelector('#fs-out').innerHTML = legend + `<br>Per-GPU memory: <b>${totalEach.toFixed(1)} GB</b>. DDP would need ${ddpEach.toFixed(1)} GB per GPU. Savings: <b>${savings}%</b>. (${P}B params, bf16+AdamW)`;
  }
  container.querySelectorAll('input,select').forEach(el => el.addEventListener('input', render));
  render();
});
