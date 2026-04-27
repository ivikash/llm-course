// viz/quantization.js
// Show fp32 -> fp16 -> bf16 -> int8 -> int4 quantization effects on a weight.

registerViz('quantization', function (container) {
  container.innerHTML = `
    <p class="viz-title">Quantization: squeezing weights into fewer bits</p>
    <p class="viz-sub">Same weight matrix, different precisions. Lower bits = smaller model, more error.</p>
    <div class="viz-controls">
      <label>Precision:
        <select id="qz-p">
          <option value="fp32" selected>FP32 (baseline)</option>
          <option value="fp16">FP16 (half)</option>
          <option value="bf16">BF16 (brain float)</option>
          <option value="int8">INT8</option>
          <option value="int4">INT4 (GPTQ/AWQ)</option>
          <option value="binary">1-bit (BitNet)</option>
        </select>
      </label>
    </div>
    <div style="display:flex;gap:16px;margin-top:12px;flex-wrap:wrap">
      <div style="flex:1;min-width:280px">
        <div style="font-size:11px;color:#6b7280;margin-bottom:4px">Original (fp32)</div>
        <canvas id="qz-orig" width="200" height="200" style="background:#fff;border:1px solid #d1d5db;border-radius:4px;image-rendering:pixelated"></canvas>
      </div>
      <div style="flex:1;min-width:280px">
        <div style="font-size:11px;color:#6b7280;margin-bottom:4px">Quantized</div>
        <canvas id="qz-q" width="200" height="200" style="background:#fff;border:1px solid #d1d5db;border-radius:4px;image-rendering:pixelated"></canvas>
      </div>
      <div style="flex:1;min-width:240px">
        <div style="font-size:11px;color:#6b7280;margin-bottom:4px">Error (|orig - quant|)</div>
        <canvas id="qz-err" width="200" height="200" style="background:#fff;border:1px solid #d1d5db;border-radius:4px;image-rendering:pixelated"></canvas>
      </div>
    </div>
    <p class="viz-readout" id="qz-out"></p>
  `;

  // Generate deterministic weight matrix
  const N = 40;
  const W = [];
  for (let i = 0; i < N; i++) {
    const row = [];
    for (let j = 0; j < N; j++) {
      row.push((Math.sin(i * 0.3) * Math.cos(j * 0.4) + Math.sin(i * j * 0.02)) * 0.5);
    }
    W.push(row);
  }

  function quantize(v, precision) {
    if (precision === 'fp32') return v;
    if (precision === 'fp16') return Math.round(v * 2048) / 2048;
    if (precision === 'bf16') return Math.round(v * 128) / 128;
    if (precision === 'int8') {
      const scale = 127 / 1.0;
      return Math.round(v * scale) / scale;
    }
    if (precision === 'int4') {
      const scale = 7 / 1.0;
      return Math.round(v * scale) / scale;
    }
    if (precision === 'binary') return Math.sign(v) * 0.5;
    return v;
  }

  function draw(cvs, mat) {
    const ctx = cvs.getContext('2d');
    const cell = 200 / N;
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const v = mat[i][j];
        const intensity = Math.floor(128 + v * 127);
        const c = Math.max(0, Math.min(255, intensity));
        ctx.fillStyle = `rgb(${c},${c},${255 - c})`;
        ctx.fillRect(j * cell, i * cell, cell, cell);
      }
    }
  }
  function drawErr(cvs, mat) {
    const ctx = cvs.getContext('2d');
    const cell = 200 / N;
    let max = 0;
    for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) max = Math.max(max, Math.abs(mat[i][j]));
    for (let i = 0; i < N; i++) {
      for (let j = 0; j < N; j++) {
        const v = Math.abs(mat[i][j]) / (max || 1);
        const intensity = Math.floor(v * 255);
        ctx.fillStyle = `rgb(${intensity},0,0)`;
        ctx.fillRect(j * cell, i * cell, cell, cell);
      }
    }
  }

  function render() {
    const p = container.querySelector('#qz-p').value;
    const Q = W.map(row => row.map(v => quantize(v, p)));
    const E = W.map((row, i) => row.map((v, j) => v - Q[i][j]));
    draw(container.querySelector('#qz-orig'), W);
    draw(container.querySelector('#qz-q'), Q);
    drawErr(container.querySelector('#qz-err'), E);

    const bits = { fp32: 32, fp16: 16, bf16: 16, int8: 8, int4: 4, binary: 1 }[p];
    const mse = E.flat().reduce((s, v) => s + v * v, 0) / (N * N);
    const modelSizeGb = (70 * bits / 8).toFixed(1);
    container.querySelector('#qz-out').innerHTML =
      `${bits} bits/weight. MSE vs fp32: <b>${mse.toExponential(2)}</b>. ` +
      `A 70B model at this precision: <b>${modelSizeGb} GB</b>. ` +
      `Llama-3-70B at int4 (~35 GB) fits on a single H100; at fp16 (140 GB) it doesn't.`;
  }
  container.querySelector('#qz-p').addEventListener('change', render);
  render();
});
