// viz/cost_calculator.js
// Cloud training cost calculator. User picks N and D; tool shows
// compute needed, time on common hardware, dollar cost.

registerViz('cost_calculator', function (container) {
  container.innerHTML = `
    <p class="viz-title">Cloud training cost calculator</p>
    <p class="viz-sub">How much will your training run cost? Real 2026 prices.</p>
    <div class="viz-controls">
      <label>Model params: <input id="cc-n" type="range" min="0.1" max="400" step="0.1" value="7"><span id="cc-n-out">7.0B</span></label>
      <label>Tokens (B): <input id="cc-d" type="range" min="1" max="15000" step="10" value="140"><span id="cc-d-out">140B</span></label>
      <label>MFU: <input id="cc-m" type="range" min="20" max="70" step="1" value="45"><span id="cc-m-out">45%</span></label>
    </div>
    <table style="margin:16px auto;border-collapse:collapse;font-size:13px">
      <thead><tr style="background:#f3f4f6"><th style="padding:8px 12px;text-align:left">Hardware</th><th>bf16 TFLOPS</th><th>GPUs</th><th>Time</th><th>Cost @ $/GPU/hr</th><th>Total $</th></tr></thead>
      <tbody id="cc-rows"></tbody>
    </table>
    <p class="viz-readout" id="cc-out"></p>
  `;
  const hardware = [
    { name: 'V100 (Pascal era)',  tflops: 125,  gpus: 8, price: 1.5 },
    { name: 'A100-40GB (Ampere)', tflops: 312,  gpus: 8, price: 2.0 },
    { name: 'A100-80GB',          tflops: 312,  gpus: 8, price: 2.5 },
    { name: 'H100-80GB (Hopper)',  tflops: 989,  gpus: 8, price: 3.0 },
    { name: 'H100 cluster',        tflops: 989,  gpus: 256, price: 3.0 },
    { name: 'H100 cluster',        tflops: 989,  gpus: 1024, price: 3.0 },
  ];
  function render() {
    const N = parseFloat(container.querySelector('#cc-n').value) * 1e9;
    const D = parseFloat(container.querySelector('#cc-d').value) * 1e9;
    const mfu = parseFloat(container.querySelector('#cc-m').value) / 100;
    container.querySelector('#cc-n-out').textContent = (N/1e9).toFixed(1) + 'B';
    container.querySelector('#cc-d-out').textContent = (D/1e9).toFixed(0) + 'B';
    container.querySelector('#cc-m-out').textContent = (mfu*100).toFixed(0) + '%';

    const flops = 6 * N * D;
    let rows = '';
    for (const hw of hardware) {
      const totalFlops = hw.tflops * 1e12 * hw.gpus * mfu;
      const seconds = flops / totalFlops;
      const hours = seconds / 3600;
      const days = hours / 24;
      const timeStr = hours < 24
        ? `${hours.toFixed(1)} hr`
        : days < 30 ? `${days.toFixed(1)} days`
        : `${(days/30).toFixed(1)} months`;
      const cost = hw.price * hw.gpus * hours;
      rows += `<tr>
        <td style="padding:8px 12px;border-top:1px solid #e5e7eb">${hw.name}</td>
        <td style="padding:8px 12px;border-top:1px solid #e5e7eb;text-align:right">${hw.tflops}</td>
        <td style="padding:8px 12px;border-top:1px solid #e5e7eb;text-align:right">${hw.gpus}</td>
        <td style="padding:8px 12px;border-top:1px solid #e5e7eb;text-align:right">${timeStr}</td>
        <td style="padding:8px 12px;border-top:1px solid #e5e7eb;text-align:right">$${hw.price.toFixed(2)}</td>
        <td style="padding:8px 12px;border-top:1px solid #e5e7eb;text-align:right;font-weight:bold">$${cost.toLocaleString(undefined, {maximumFractionDigits: 0})}</td>
      </tr>`;
    }
    container.querySelector('#cc-rows').innerHTML = rows;
    container.querySelector('#cc-out').textContent =
      `Total training FLOPs = 6 × ${(N/1e9).toFixed(1)}B × ${(D/1e9).toFixed(0)}B = ${flops.toExponential(1)}. ` +
      `Using Chinchilla-optimal (D ≈ 20×N), ${(N/1e9).toFixed(1)}B model should see ≈ ${((20*N)/1e9).toFixed(0)}B tokens.`;
  }
  container.querySelectorAll('input').forEach(el => el.addEventListener('input', render));
  render();
});
