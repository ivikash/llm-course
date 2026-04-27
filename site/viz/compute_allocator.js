// viz/compute_allocator.js
// Quarterly compute budget planner. Drag sliders to allocate
// GPU-hours across exploration / validation / scaled / buffer.

registerViz('compute_allocator', function (container) {
  container.innerHTML = `
    <p class="viz-title">Quarterly compute budget</p>
    <p class="viz-sub">A team leader's tool. Allocate GPU-hours across project types. Does it add up to what you have?</p>
    <div class="viz-controls">
      <label>Total GPU-hours: <input id="ca-t" type="range" min="1000" max="100000" step="500" value="17280"><span id="ca-t-out">17,280</span></label>
    </div>
    <div id="ca-rows" style="display:flex;flex-direction:column;gap:8px;margin-top:12px"></div>
    <div id="ca-bar" style="margin-top:16px;height:40px;border-radius:4px;border:1px solid #d1d5db;overflow:hidden;display:flex"></div>
    <p class="viz-readout" id="ca-out"></p>
  `;
  const items = [
    { key: 'unavoidable', name: 'Unavoidable (serving, eval)', pct: 17, color: '#6b7280', desc: 'ongoing inference + eval suite' },
    { key: 'explore',     name: 'Exploration (many small runs)', pct: 29, color: '#a78bfa', desc: '30-60 small experiments' },
    { key: 'validate',    name: 'Validation (promising ideas)', pct: 29, color: '#60a5fa', desc: '5-10 medium runs' },
    { key: 'scaled',      name: 'Scaled experiments (big bets)', pct: 11, color: '#f59e0b', desc: '1-2 big runs' },
    { key: 'buffer',      name: 'Buffer (unplanned)', pct: 14, color: '#fca5a5', desc: 'keep some headroom' },
  ];
  function render() {
    const total = +container.querySelector('#ca-t').value;
    container.querySelector('#ca-t-out').textContent = total.toLocaleString();
    let rows = '';
    items.forEach(it => {
      rows += `<div style="display:flex;align-items:center;gap:8px;font-size:13px">
        <div style="width:220px">${it.name}</div>
        <input data-key="${it.key}" type="range" min="0" max="80" value="${it.pct}" style="flex:1">
        <div style="width:40px;text-align:right;font-family:ui-monospace,monospace" data-val="${it.key}">${it.pct}%</div>
        <div style="width:100px;text-align:right;color:#6b7280" data-hours="${it.key}">${Math.floor(total*it.pct/100).toLocaleString()} hr</div>
      </div>`;
    });
    container.querySelector('#ca-rows').innerHTML = rows;

    let bar = '';
    let cum = 0;
    items.forEach(it => {
      bar += `<div style="width:${it.pct}%;height:100%;background:${it.color};display:flex;align-items:center;justify-content:center;color:#fff;font-size:11px;overflow:hidden" title="${it.name}: ${it.pct}%">${it.pct > 8 ? it.pct + '%' : ''}</div>`;
      cum += it.pct;
    });
    container.querySelector('#ca-bar').innerHTML = bar;

    const sum = items.reduce((s, it) => s + it.pct, 0);
    const diag = sum === 100 ? '✓ Balanced' : sum > 100 ? `⚠️ Over-committed by ${sum - 100}%` : `Under-committed by ${100 - sum}%`;
    container.querySelector('#ca-out').textContent =
      `Total allocated: ${sum}% (${(total*sum/100).toLocaleString()} GPU-hours). ${diag}. ` +
      `Convention: unavoidable 15-20%, exploration 25-35%, validation 25-30%, scaled 10-15%, buffer 10-15%.`;

    // wire up sliders
    container.querySelectorAll('input[data-key]').forEach(el => {
      el.addEventListener('input', () => {
        const it = items.find(i => i.key === el.dataset.key);
        it.pct = +el.value;
        render();
      });
    });
  }
  container.querySelector('#ca-t').addEventListener('input', render);
  render();
});
