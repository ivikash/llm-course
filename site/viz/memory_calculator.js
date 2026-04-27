// viz/memory_calculator.js
// GPU memory budget interactive. User picks model size, precision,
// batch; tool shows where memory goes and whether it OOMs.

registerViz('memory_calculator', function (container) {
  container.innerHTML = `
    <p class="viz-title">GPU memory calculator</p>
    <p class="viz-sub">Will this model fit? Answers the question for training, live.</p>
    <div class="viz-controls">
      <label>Model params: <input id="mc-p" type="range" min="0.1" max="70" step="0.1" value="7"><span id="mc-p-out">7.0B</span></label>
      <label>GPU VRAM: <select id="mc-g">
        <option>A100-40GB</option><option selected>A100-80GB</option><option>H100-80GB</option><option>H100-141GB</option><option>4090-24GB</option>
      </select></label>
      <label>Precision: <select id="mc-pr">
        <option>fp32</option><option selected>bf16+AdamW</option><option>bf16+8-bit-adam</option><option>fp16+loss-scaler</option>
      </select></label>
      <label>Batch tokens: <input id="mc-bt" type="range" min="512" max="131072" step="512" value="8192"><span id="mc-bt-out">8192</span></label>
    </div>
    <div id="mc-bars" style="display:flex;flex-direction:column;gap:6px;margin-top:16px"></div>
    <p class="viz-readout" id="mc-out"></p>
  `;
  function render() {
    const P = parseFloat(container.querySelector('#mc-p').value) * 1e9;
    const gpu = container.querySelector('#mc-g').value;
    const pr = container.querySelector('#mc-pr').value;
    const bt = +container.querySelector('#mc-bt').value;
    container.querySelector('#mc-p-out').textContent = (P/1e9).toFixed(1) + 'B';
    container.querySelector('#mc-bt-out').textContent = bt.toLocaleString();
    const vramGB = { 'A100-40GB': 40, 'A100-80GB': 80, 'H100-80GB': 80, 'H100-141GB': 141, '4090-24GB': 24 }[gpu];
    const vram = vramGB * 1e9;

    // bytes per param
    const bytes = {
      'fp32': { master: 4, working: 0, grad: 4, m: 4, v: 4 },
      'bf16+AdamW': { master: 4, working: 2, grad: 2, m: 4, v: 4 },
      'bf16+8-bit-adam': { master: 4, working: 2, grad: 2, m: 1, v: 1 },
      'fp16+loss-scaler': { master: 4, working: 2, grad: 2, m: 4, v: 4 },
    }[pr];
    const perP = bytes.master + bytes.working + bytes.grad + bytes.m + bytes.v;
    const modelMem = P * perP;
    // activations: rough estimate ~ batch_tokens × hidden_dim × layers × 16 bytes
    // approximate hidden_dim from params: h ≈ sqrt(P/24) (back-of-envelope)
    const h = Math.max(768, Math.sqrt(P / 24));
    const L = Math.max(12, 12 * Math.cbrt(P / 100e6));
    const actMem = bt * h * L * 16;
    const total = modelMem + actMem;
    const fits = total < vram;

    const items = [
      { name: 'Model weights (fp32 master + bf16 working)', bytes: P * (bytes.master + bytes.working), color: '#3b82f6' },
      { name: 'Gradients', bytes: P * bytes.grad, color: '#f59e0b' },
      { name: 'Optimizer state (m + v)', bytes: P * (bytes.m + bytes.v), color: '#10b981' },
      { name: 'Activations (batch × ~layer)', bytes: actMem, color: '#8b5cf6' },
    ];
    const maxBytes = Math.max(vram, total);
    let html = `<div style="display:flex;align-items:center;gap:8px;font-size:13px;margin-bottom:6px">
      <div style="font-weight:bold">GPU VRAM limit:</div>
      <div style="flex:1;height:12px;background:#f3f4f6;border-radius:3px;position:relative;overflow:hidden">
        <div style="position:absolute;left:${(vram/maxBytes)*100}%;top:-2px;width:2px;height:16px;background:#dc2626"></div>
      </div>
      <div style="font-family:ui-monospace,monospace">${(vram/1e9).toFixed(1)} GB</div>
    </div>`;
    items.forEach(it => {
      const pct = (it.bytes / maxBytes) * 100;
      html += `<div style="display:flex;align-items:center;gap:8px;font-size:13px">
        <div style="width:260px">${it.name}</div>
        <div style="flex:1;height:18px;background:#f3f4f6;border-radius:3px;overflow:hidden">
          <div style="width:${pct}%;height:100%;background:${it.color}"></div>
        </div>
        <div style="font-family:ui-monospace,monospace;width:70px;text-align:right">${(it.bytes/1e9).toFixed(2)} GB</div>
      </div>`;
    });
    container.querySelector('#mc-bars').innerHTML = html;
    container.querySelector('#mc-out').innerHTML =
      `<b>Total: ${(total/1e9).toFixed(1)} GB</b> — ` +
      (fits
        ? `<span style="color:#15803d">✓ FITS on ${gpu} (${(vramGB - total/1e9).toFixed(1)} GB headroom)</span>`
        : `<span style="color:#b91c1c">✗ OOM on ${gpu} — need FSDP, smaller model, or smaller batch</span>`);
  }
  container.querySelectorAll('input,select').forEach(el => el.addEventListener('input', render));
  render();
});
