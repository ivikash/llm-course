// viz/grad_accumulation.js
// Animation: big batch vs micro-batches with gradient accumulation.

registerViz('grad_accumulation', function (container) {
  container.innerHTML = `
    <p class="viz-title">Gradient accumulation: simulating a bigger batch</p>
    <p class="viz-sub">No room for batch=64 on your GPU? Run 4 micro-batches of 16, accumulate gradients, then step. Same gradient as one big batch.</p>
    <div class="viz-controls">
      <button id="ga-play">▶ Play one effective step</button>
      <label>accum steps: <input id="ga-k" type="range" min="1" max="8" value="4"><span id="ga-k-out">4</span></label>
      <label>micro batch: <input id="ga-m" type="range" min="2" max="32" value="16"><span id="ga-m-out">16</span></label>
    </div>
    <svg id="ga-svg" viewBox="0 0 700 260" width="700" height="260" style="background:#fff;border:1px solid #e5e7eb;border-radius:4px;max-width:100%"></svg>
    <p class="viz-readout" id="ga-out"></p>
  `;
  const svg = container.querySelector('#ga-svg');
  let curStep = -1;
  let zeroGrad = false;

  function render() {
    const K = +container.querySelector('#ga-k').value;
    const M = +container.querySelector('#ga-m').value;
    container.querySelector('#ga-k-out').textContent = K;
    container.querySelector('#ga-m-out').textContent = M;
    const effective = K * M;
    const boxW = 600 / K;
    let html = `<text x="20" y="20" font-size="12" fill="#374151" font-weight="bold">Effective batch = ${K} × ${M} = ${effective}</text>`;
    html += `<line x1="40" y1="140" x2="680" y2="140" stroke="#6b7280"/>`;
    for (let i = 0; i < K; i++) {
      const x = 40 + i * boxW;
      const active = i === curStep;
      const done = i < curStep;
      const bg = active ? '#fbbf24' : done ? '#bbf7d0' : '#e5e7eb';
      html += `<rect x="${x}" y="60" width="${boxW - 8}" height="70" fill="${bg}" stroke="#6b7280" rx="4"/>`;
      html += `<text x="${x + boxW/2 - 4}" y="90" font-size="12" text-anchor="middle" fill="#111">micro-batch ${i+1}</text>`;
      html += `<text x="${x + boxW/2 - 4}" y="108" font-size="11" text-anchor="middle" fill="#6b7280">fwd → loss/K → bwd</text>`;
      html += `<text x="${x + boxW/2 - 4}" y="124" font-size="10" text-anchor="middle" fill="#6b7280">(${M} samples)</text>`;
    }
    // accumulated gradient bar
    const gradPct = curStep >= 0 ? ((curStep + 1) / K) * 100 : 0;
    html += `<text x="20" y="170" font-size="11" fill="#374151">Accumulated gradient buffer:</text>`;
    html += `<rect x="40" y="180" width="600" height="24" fill="#f3f4f6" stroke="#6b7280" rx="3"/>`;
    html += `<rect x="40" y="180" width="${gradPct * 6}" height="24" fill="#2563eb"/>`;
    html += `<text x="340" y="198" font-size="12" text-anchor="middle" fill="#fff" font-weight="bold">${curStep >= 0 ? `${curStep + 1}/${K} micro-batches summed` : 'empty'}</text>`;

    // optimizer step indicator
    if (curStep === K) {
      html += `<text x="340" y="240" font-size="14" text-anchor="middle" fill="#15803d" font-weight="bold">✓ optimizer.step() → zero_grad()</text>`;
    }
    svg.innerHTML = html;
  }

  async function play() {
    const K = +container.querySelector('#ga-k').value;
    curStep = -1;
    for (let i = 0; i < K; i++) {
      curStep = i;
      render();
      await new Promise(r => setTimeout(r, 700));
    }
    curStep = K;
    render();
    await new Promise(r => setTimeout(r, 1000));
    curStep = -1;
    render();
  }

  container.querySelector('#ga-k').addEventListener('input', render);
  container.querySelector('#ga-m').addEventListener('input', render);
  container.querySelector('#ga-play').addEventListener('click', play);
  render();
  const out = container.querySelector('#ga-out');
  out.textContent = `During each micro-batch: loss is divided by K before backward, so summed gradients equal the mean gradient over effective_batch samples.`;
});
