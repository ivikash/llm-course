// viz/wandb_dashboard.js
// Simulated wandb training dashboard — live metrics streaming.

registerViz('wandb_dashboard', function (container) {
  container.innerHTML = `
    <p class="viz-title">Your training run, live (simulated wandb)</p>
    <p class="viz-sub">This is what you'll stare at for weeks. 4 metric panels updating in real time.</p>
    <div class="viz-controls">
      <button id="wb-start">▶ Start training</button>
      <button id="wb-stop">■ Stop</button>
      <button id="wb-reset">Reset</button>
      <label>speed: <input id="wb-speed" type="range" min="20" max="200" step="10" value="80"><span id="wb-speed-out">80</span>ms/step</label>
    </div>
    <div id="wb-grid" style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-top:12px"></div>
    <p class="viz-readout" id="wb-out"></p>
  `;

  const panels = [
    { key: 'train/loss',    title: 'train/loss',    color: '#2563eb', min: 0, max: 10 },
    { key: 'val/bpb',       title: 'val/bpb',       color: '#059669', min: 0.7, max: 2 },
    { key: 'train/grad_norm', title: 'train/grad_norm', color: '#f59e0b', min: 0, max: 3 },
    { key: 'train/mfu',     title: 'train/mfu',     color: '#a855f7', min: 0, max: 60 },
  ];
  let step = 0;
  const data = Object.fromEntries(panels.map(p => [p.key, []]));
  let timer = null;

  function simulate() {
    const maxStep = 5000;
    // train loss: decay from 10 to ~2.5
    const warmup = 200;
    let tl;
    if (step < warmup) tl = 10 - 5 * step / warmup;
    else tl = 2.5 + 2.5 * Math.exp(-(step - warmup) / 600);
    tl += (Math.random() - 0.5) * 0.3;
    data['train/loss'].push([step, tl]);
    // val bpb: every 500 steps
    if (step % 500 === 0) {
      const vb = 0.9 + (tl - 2.5) * 0.12 + (Math.random() - 0.5) * 0.02;
      data['val/bpb'].push([step, vb]);
    }
    // grad norm: centered around 1
    const gn = 0.8 + 0.4 * Math.random() + (step % 1000 < 50 ? Math.random() : 0);
    data['train/grad_norm'].push([step, gn]);
    // MFU: ramps up then steady
    const mfu = step < 200 ? step / 200 * 45 : 44 + 2 * Math.sin(step / 50) + Math.random();
    data['train/mfu'].push([step, mfu]);
    step++;
  }

  function render() {
    const grid = container.querySelector('#wb-grid');
    grid.innerHTML = panels.map(p => {
      const pts = data[p.key];
      const W = 300, H = 120;
      let path = '';
      if (pts.length > 1) {
        for (let i = 0; i < pts.length; i++) {
          const [s, v] = pts[i];
          const x = 20 + (s / 5000) * (W - 30);
          const y = H - 10 - ((v - p.min) / (p.max - p.min)) * (H - 20);
          path += (path ? ' L ' : 'M ') + x.toFixed(1) + ' ' + y.toFixed(1);
        }
      }
      const lastV = pts.length ? pts[pts.length - 1][1] : NaN;
      return `<div style="border:1px solid #e5e7eb;border-radius:6px;padding:8px;background:#fff">
        <div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:4px">
          <span style="color:${p.color};font-weight:bold">${p.title}</span>
          <span style="font-family:ui-monospace,monospace;color:#6b7280">${isNaN(lastV) ? '—' : lastV.toFixed(3)}</span>
        </div>
        <svg viewBox="0 0 ${W} ${H}" width="100%" height="${H}">
          <line x1="20" y1="${H-10}" x2="${W-10}" y2="${H-10}" stroke="#e5e7eb"/>
          <line x1="20" y1="10" x2="20" y2="${H-10}" stroke="#e5e7eb"/>
          <text x="18" y="${H-10}" font-size="9" fill="#9ca3af" text-anchor="end">${p.min}</text>
          <text x="18" y="15" font-size="9" fill="#9ca3af" text-anchor="end">${p.max}</text>
          <path d="${path}" fill="none" stroke="${p.color}" stroke-width="1.5"/>
        </svg>
      </div>`;
    }).join('');
    container.querySelector('#wb-out').textContent = `step ${step} / 5000 — ` + (timer ? 'training...' : 'paused');
  }

  function start() {
    if (timer) return;
    const speed = +container.querySelector('#wb-speed').value;
    timer = setInterval(() => {
      simulate();
      if (step % 5 === 0) render();
      if (step >= 5000) { clearInterval(timer); timer = null; render(); }
    }, speed);
  }

  container.querySelector('#wb-speed').addEventListener('input', e => {
    container.querySelector('#wb-speed-out').textContent = e.target.value;
    if (timer) { clearInterval(timer); timer = null; start(); }
  });
  container.querySelector('#wb-start').addEventListener('click', start);
  container.querySelector('#wb-stop').addEventListener('click', () => { if (timer) { clearInterval(timer); timer = null; render(); } });
  container.querySelector('#wb-reset').addEventListener('click', () => {
    if (timer) { clearInterval(timer); timer = null; }
    step = 0;
    for (const p of panels) data[p.key] = [];
    render();
  });
  render();
});
