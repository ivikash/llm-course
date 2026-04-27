// viz/layernorm_residual.js
// Compare pre-norm vs post-norm block. Show gradient magnitudes
// decaying through 20 stacked blocks (pre-norm stays healthy, post-norm vanishes).

registerViz('layernorm_residual', function (container) {
  container.innerHTML = `
    <p class="viz-title">Pre-norm vs post-norm: why deep transformers train</p>
    <p class="viz-sub">Both architectures work for shallow models. Only pre-norm is stable at depth. Here: gradient magnitudes through 20 stacked blocks.</p>
    <div class="viz-controls">
      <label>Depth: <input id="lnr-d" type="range" min="2" max="48" value="20"><span id="lnr-d-out">20</span></label>
    </div>
    <div style="display:flex;gap:16px;flex-wrap:wrap;justify-content:center;margin-top:12px">
      <div>
        <div style="text-align:center;color:#dc2626;font-weight:bold;margin-bottom:4px">Post-norm (original 2017)</div>
        <svg id="lnr-post" viewBox="0 0 340 220" width="340" height="220" style="background:#fff;border:1px solid #e5e7eb;border-radius:4px"></svg>
      </div>
      <div>
        <div style="text-align:center;color:#10b981;font-weight:bold;margin-bottom:4px">Pre-norm (GPT-2+)</div>
        <svg id="lnr-pre" viewBox="0 0 340 220" width="340" height="220" style="background:#fff;border:1px solid #e5e7eb;border-radius:4px"></svg>
      </div>
    </div>
    <p class="viz-readout" id="lnr-out"></p>
  `;
  function drawFlow(svgId, variant) {
    const depth = +container.querySelector('#lnr-d').value;
    const svg = container.querySelector('#' + svgId);
    const W = 340, H = 220;
    const padX = 30, padY = 20;
    const plotW = W - 2*padX, plotH = H - 2*padY;
    // Simulate gradient magnitude through blocks (toy model)
    // Post-norm: LN on skip path attenuates gradient each layer: *0.92^L
    // Pre-norm: identity skip, grad stays ~1 with some noise
    const grads = [];
    for (let i = 0; i < depth; i++) {
      if (variant === 'post') grads.push(Math.pow(0.88, i) * (0.9 + 0.2*Math.random()));
      else grads.push(0.8 + 0.4*Math.random());
    }
    // plot
    const barW = plotW / depth;
    let html = `<line x1="${padX}" y1="${padY + plotH}" x2="${padX + plotW}" y2="${padY + plotH}" stroke="#6b7280"/>`;
    html += `<line x1="${padX}" y1="${padY}" x2="${padX}" y2="${padY + plotH}" stroke="#6b7280"/>`;
    html += `<text x="10" y="${padY + plotH/2}" font-size="10" fill="#6b7280" transform="rotate(-90 10 ${padY+plotH/2})">grad mag</text>`;
    html += `<text x="${W/2}" y="${H - 4}" font-size="10" fill="#6b7280" text-anchor="middle">layer depth</text>`;
    grads.forEach((g, i) => {
      const x = padX + i * barW;
      const h = Math.min(1, g) * plotH;
      const color = variant === 'post' ? '#dc2626' : '#10b981';
      html += `<rect x="${x + 1}" y="${padY + plotH - h}" width="${barW - 2}" height="${h}" fill="${color}" opacity="0.7"/>`;
    });
    svg.innerHTML = html;
    return grads;
  }
  function render() {
    const depth = +container.querySelector('#lnr-d').value;
    container.querySelector('#lnr-d-out').textContent = depth;
    const post = drawFlow('lnr-post', 'post');
    const pre  = drawFlow('lnr-pre', 'pre');
    const postMin = Math.min(...post).toFixed(3);
    const preMin = Math.min(...pre).toFixed(3);
    container.querySelector('#lnr-out').innerHTML =
      `At depth ${depth}: post-norm gradients at layer 0 ≈ <b style="color:#dc2626">${postMin}</b> (vanishing!). ` +
      `Pre-norm stays ~<b style="color:#10b981">${preMin}</b>. ` +
      `Below ~10 layers, both train. Beyond ~24, only pre-norm is reliable. Why every modern LLM uses pre-norm.`;
  }
  container.querySelector('#lnr-d').addEventListener('input', render);
  render();
});
