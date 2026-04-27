// viz/throughput_calc.js
// Inference throughput / latency calculator: memory-bound vs compute-bound.

registerViz('throughput_calc', function (container) {
  container.innerHTML = `
    <p class="viz-title">Inference throughput & latency calculator</p>
    <p class="viz-sub">Is your workload memory-bound or compute-bound? Every deployment decision depends on this.</p>
    <div class="viz-controls" style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
      <label>Model params: <input id="tc-p" type="range" min="1" max="180" value="8"><span id="tc-p-out">8B</span></label>
      <label>GPU:
        <select id="tc-g">
          <option value="a100" selected>A100 (1555 GB/s, 312 TFLOPS)</option>
          <option value="h100">H100 (3350 GB/s, 989 TFLOPS)</option>
          <option value="b200">B200 (8000 GB/s, 4500 TFLOPS)</option>
          <option value="l40s">L40S (864 GB/s, 362 TFLOPS)</option>
        </select>
      </label>
      <label>Batch size: <input id="tc-b" type="range" min="1" max="256" value="1"><span id="tc-b-out">1</span></label>
      <label>Precision:
        <select id="tc-pr">
          <option value="fp16" selected>FP16 (2 bytes/param)</option>
          <option value="int8">INT8 (1 byte/param)</option>
          <option value="int4">INT4 (0.5 byte/param)</option>
        </select>
      </label>
      <label>Context length: <input id="tc-c" type="range" min="128" max="16384" step="128" value="2048"><span id="tc-c-out">2048</span></label>
      <label>Output tokens: <input id="tc-o" type="range" min="10" max="4096" step="10" value="500"><span id="tc-o-out">500</span></label>
    </div>
    <div id="tc-out" style="margin-top:14px;padding:12px;background:#f3f4f6;border-radius:6px;font-size:13px;line-height:1.7"></div>
  `;

  const gpus = {
    a100: { bw: 1555, tflops: 312, mem: 80 },
    h100: { bw: 3350, tflops: 989, mem: 80 },
    b200: { bw: 8000, tflops: 4500, mem: 192 },
    l40s: { bw: 864, tflops: 362, mem: 48 },
  };

  function render() {
    const P = +container.querySelector('#tc-p').value;
    const B = +container.querySelector('#tc-b').value;
    const C = +container.querySelector('#tc-c').value;
    const OUT = +container.querySelector('#tc-o').value;
    const gpu = gpus[container.querySelector('#tc-g').value];
    const pr = container.querySelector('#tc-pr').value;
    const bpp = { fp16: 2, int8: 1, int4: 0.5 }[pr];

    container.querySelector('#tc-p-out').textContent = P + 'B';
    container.querySelector('#tc-b-out').textContent = B;
    container.querySelector('#tc-c-out').textContent = C;
    container.querySelector('#tc-o-out').textContent = OUT;

    const modelGb = P * bpp;
    // KV cache: 2 bytes per element, 2 (K+V), d_head * n_heads = d_model, avg layers proportional to sqrt
    const layers = Math.ceil(Math.pow(P, 0.5) * 8);
    const d = Math.ceil(Math.pow(P * 1e9, 1/3) * 8 / layers) * layers;  // rough
    const kvGb = 2 * 2 * layers * d * C * B / 1e9;
    const totalMem = modelGb + kvGb;
    const fits = totalMem < gpu.mem;

    // Decode: memory-bound. Per token: must read all weights + KV
    const decodeMs = (modelGb + kvGb) * 1000 / gpu.bw;  // ms
    const decodeTps = B / (decodeMs / 1000);
    // Prefill: compute-bound. Flops ~ 2*P*C*B
    const prefillS = 2 * P * 1e9 * C * B / (gpu.tflops * 1e12);
    const prefillMs = prefillS * 1000;
    // Total request latency
    const totalLatency = prefillMs + decodeMs * OUT;

    let html = '';
    html += `<b>Memory:</b> ${modelGb.toFixed(1)} GB (weights) + ${kvGb.toFixed(1)} GB (KV cache) = ${totalMem.toFixed(1)} GB / ${gpu.mem} GB ` +
            (fits ? '<span style="color:#15803d">✓ fits</span>' : '<span style="color:#dc2626">✗ OOM!</span>') + '<br>';
    html += `<b>Decode speed:</b> ${decodeMs.toFixed(1)} ms/token, ${decodeTps.toFixed(0)} tokens/s (batch ${B})<br>`;
    html += `<b>Prefill latency:</b> ${prefillMs.toFixed(0)} ms (${C} tokens)<br>`;
    html += `<b>Total request:</b> ${(totalLatency/1000).toFixed(2)} s (prefill + ${OUT} decode tokens)<br>`;
    html += `<b>Bound by:</b> decoding is <span style="color:${decodeMs > 50 ? '#dc2626' : '#15803d'}">memory-bound</span> (reading ${modelGb.toFixed(0)}+GB per token). `;
    html += `Batching helps: at B=${B}, each GPU cycle produces ${B} tokens instead of 1.<br>`;
    if (B < 16 && fits) html += `<i>Tip: try increasing batch size for ${(16/B).toFixed(1)}× more throughput.</i>`;
    if (!fits) html += `<i style="color:#dc2626">Tip: try INT4 quantization or a bigger GPU.</i>`;
    container.querySelector('#tc-out').innerHTML = html;
  }
  container.querySelectorAll('input,select').forEach(el => el.addEventListener('input', render));
  render();
});
