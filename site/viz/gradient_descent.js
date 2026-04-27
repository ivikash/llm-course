// viz/gradient_descent.js
// Live gradient descent on a 1D quadratic loss.
// Shows: ball rolling down a curve. Sliders for learning rate + start.
// Critical: shows lr too big = diverges, too small = crawls.

registerViz('gradient_descent', function (container) {
  container.innerHTML = `
    <p class="viz-title">Gradient descent, live</p>
    <p class="viz-sub">A ball rolling down L(w) = (w − 5)². Adjust learning rate and start. Watch it converge, crawl, or explode.</p>
    <div class="viz-controls">
      <label>Learning rate: <input id="gd-lr" type="range" min="0.01" max="1.2" step="0.01" value="0.1"><span id="gd-lr-out">0.10</span></label>
      <label>Start w: <input id="gd-w0" type="range" min="-10" max="15" step="0.5" value="-5"><span id="gd-w0-out">-5</span></label>
      <button id="gd-run">Run</button>
      <button id="gd-step">Single step</button>
    </div>
    <svg id="gd-svg" viewBox="0 0 600 300" width="600" height="300" style="border:1px solid #e5e7eb;border-radius:4px;background:#fff;max-width:100%"></svg>
    <p class="viz-readout" id="gd-out"></p>
  `;

  const svg = container.querySelector('#gd-svg');
  const lrEl = container.querySelector('#gd-lr');
  const w0El = container.querySelector('#gd-w0');
  const lrOut = container.querySelector('#gd-lr-out');
  const w0Out = container.querySelector('#gd-w0-out');
  const out = container.querySelector('#gd-out');
  const runBtn = container.querySelector('#gd-run');
  const stepBtn = container.querySelector('#gd-step');

  // Loss: L(w) = (w-5)^2, dL/dw = 2*(w-5)
  const L = w => (w - 5) ** 2;
  const dL = w => 2 * (w - 5);

  let w = parseFloat(w0El.value);
  let history = [w];

  function wToX(w) { return 30 + (w + 10) * (540 / 25); } // w in [-10,15] → [30,570]
  function lToY(l) { return 260 - Math.min(l, 200) * 1.0; } // loss up to 200

  function drawCurve() {
    let path = '';
    for (let wx = -10; wx <= 15; wx += 0.2) {
      const x = wToX(wx), y = lToY(L(wx));
      path += (path ? ' L ' : 'M ') + x.toFixed(1) + ' ' + y.toFixed(1);
    }
    return `<path d="${path}" fill="none" stroke="#9ca3af" stroke-width="2"/>`;
  }

  function render() {
    const ball = `<circle cx="${wToX(w)}" cy="${lToY(L(w))}" r="8" fill="#ef4444"/>`;
    const trail = history.map((hw, i) => {
      const alpha = 0.2 + 0.8 * (i / Math.max(1, history.length - 1));
      return `<circle cx="${wToX(hw)}" cy="${lToY(L(hw))}" r="3" fill="rgba(37,99,235,${alpha.toFixed(2)})"/>`;
    }).join('');
    // minimum marker
    const minLine = `<line x1="${wToX(5)}" y1="0" x2="${wToX(5)}" y2="300" stroke="#10b981" stroke-dasharray="3,3" opacity="0.5"/>`;
    const gridX = Array.from({ length: 6 }, (_, i) => -10 + i * 5).map(v =>
      `<text x="${wToX(v)}" y="290" font-size="10" fill="#6b7280" text-anchor="middle">w=${v}</text>`).join('');
    svg.innerHTML = `
      ${drawCurve()}
      ${minLine}
      <text x="${wToX(5) + 5}" y="12" font-size="11" fill="#10b981">minimum</text>
      ${trail}
      ${ball}
      ${gridX}
    `;
    out.innerHTML = `w = ${w.toFixed(3)}   loss = ${L(w).toFixed(3)}   dL/dw = ${dL(w).toFixed(3)}\n` +
      `steps taken: ${history.length - 1}` +
      (Math.abs(w) > 1e5 ? '   <span style="color:#b91c1c"><b>DIVERGED</b></span>' :
       Math.abs(L(w)) < 1e-4 ? '   <span style="color:#15803d"><b>CONVERGED</b></span>' : '');
  }

  function step() {
    const lr = parseFloat(lrEl.value);
    w = w - lr * dL(w);
    history.push(w);
    if (history.length > 200) history.shift();
    render();
  }

  let runHandle = null;
  function stopRun() {
    if (runHandle) { clearInterval(runHandle); runHandle = null; runBtn.textContent = 'Run'; }
  }
  function run() {
    if (runHandle) { stopRun(); return; }
    runBtn.textContent = 'Stop';
    runHandle = setInterval(() => {
      step();
      if (Math.abs(w) > 1e5 || Math.abs(L(w)) < 1e-5) stopRun();
    }, 80);
  }

  lrEl.addEventListener('input', () => { lrOut.textContent = parseFloat(lrEl.value).toFixed(2); });
  w0El.addEventListener('input', () => {
    w0Out.textContent = w0El.value;
    stopRun();
    w = parseFloat(w0El.value); history = [w]; render();
  });
  runBtn.addEventListener('click', run);
  stepBtn.addEventListener('click', () => { stopRun(); step(); });

  render();
});
