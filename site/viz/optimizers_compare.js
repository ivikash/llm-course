// viz/optimizers_compare.js
// Four optimizers (SGD, SGD+momentum, Adam, AdamW) on the same 2D loss surface.
// User presses "Run" and watches the paths.

registerViz('optimizers_compare', function (container) {
  container.innerHTML = `
    <p class="viz-title">Optimizers on a 2D loss surface</p>
    <p class="viz-sub">Same starting point. Four optimizers race down the Rosenbrock-ish valley. Watch which finds the minimum fastest.</p>
    <div class="viz-controls">
      <label>Learning rate: <input id="op-lr" type="range" min="0.001" max="0.1" step="0.001" value="0.01"><span id="op-lr-out">0.01</span></label>
      <button id="op-run">Run</button>
      <button id="op-reset">Reset</button>
    </div>
    <svg id="op-svg" viewBox="0 0 600 400" width="600" height="400" style="background:#fff;border:1px solid #e5e7eb;border-radius:4px;max-width:100%"></svg>
    <p class="viz-readout" id="op-out"></p>
  `;
  const svg = container.querySelector('#op-svg');
  const out = container.querySelector('#op-out');
  const lrEl = container.querySelector('#op-lr');
  const lrOut = container.querySelector('#op-lr-out');
  const runBtn = container.querySelector('#op-run');
  const resetBtn = container.querySelector('#op-reset');

  // Loss: elongated bowl. min at (3, 3).
  const L = (x, y) => 0.05 * ((x - 3) ** 2) + 0.5 * ((y - 3) ** 2);
  const dL = (x, y) => [0.1 * (x - 3), 1.0 * (y - 3)];

  // Contour data
  function drawContours() {
    const xToPx = x => 30 + (x + 2) * (540 / 10);
    const yToPx = y => 370 - (y + 2) * (340 / 10);
    const lines = [];
    // fill gradient background
    const levels = [0.5, 1.5, 3, 6, 10, 15];
    // iso-contour via grid + marching
    const step = 0.3;
    for (const lv of levels) {
      // sample points roughly on the isoline
      let pts = [];
      for (let a = 0; a < Math.PI * 2; a += 0.01) {
        // parametric: ellipse aligned with axes
        const r = Math.sqrt(lv);
        const x = 3 + (r / Math.sqrt(0.05)) * Math.cos(a);
        const y = 3 + (r / Math.sqrt(0.5)) * Math.sin(a);
        pts.push([xToPx(x), yToPx(y)]);
      }
      lines.push(`<path d="M ${pts.map(p => p.join(' ')).join(' L ')}" fill="none" stroke="#e5e7eb" stroke-width="1"/>`);
    }
    return lines.join('');
  }

  const startX = -0.5, startY = -0.5;
  const maxSteps = 300;

  function makeSGD(lr) {
    let x = startX, y = startY;
    const path = [[x, y]];
    return {
      step() { const [gx, gy] = dL(x, y); x -= lr * gx; y -= lr * gy; path.push([x, y]); },
      path() { return path; }
    };
  }
  function makeMom(lr) {
    let x = startX, y = startY; let vx = 0, vy = 0;
    const path = [[x, y]];
    return {
      step() { const [gx, gy] = dL(x, y);
        vx = 0.9 * vx + gx; vy = 0.9 * vy + gy;
        x -= lr * vx; y -= lr * vy; path.push([x, y]); },
      path() { return path; }
    };
  }
  function makeAdam(lr) {
    let x = startX, y = startY; let mx = 0, my = 0, vx = 0, vy = 0; let t = 0;
    const b1 = 0.9, b2 = 0.999, eps = 1e-8;
    const path = [[x, y]];
    return {
      step() { t++;
        const [gx, gy] = dL(x, y);
        mx = b1 * mx + (1 - b1) * gx; my = b1 * my + (1 - b1) * gy;
        vx = b2 * vx + (1 - b2) * gx * gx; vy = b2 * vy + (1 - b2) * gy * gy;
        const mxh = mx / (1 - b1 ** t), myh = my / (1 - b1 ** t);
        const vxh = vx / (1 - b2 ** t), vyh = vy / (1 - b2 ** t);
        x -= lr * mxh / (Math.sqrt(vxh) + eps);
        y -= lr * myh / (Math.sqrt(vyh) + eps);
        path.push([x, y]);
      },
      path() { return path; }
    };
  }

  const xToPx = x => 30 + (x + 2) * (540 / 10);
  const yToPx = y => 370 - (y + 2) * (340 / 10);

  let opts, stepCount = 0, intervalId = null;

  function reset() {
    stepCount = 0;
    const lr = parseFloat(lrEl.value);
    opts = [
      { name: 'SGD',          color: '#ef4444', opt: makeSGD(lr) },
      { name: 'SGD+momentum', color: '#f59e0b', opt: makeMom(lr) },
      { name: 'Adam',         color: '#2563eb', opt: makeAdam(lr) },
    ];
    render();
  }

  function render() {
    let pathsHtml = '';
    let legendHtml = '';
    opts.forEach((o, i) => {
      const pts = o.opt.path();
      const pxs = pts.map(p => `${xToPx(p[0]).toFixed(1)} ${yToPx(p[1]).toFixed(1)}`);
      pathsHtml += `<path d="M ${pxs.join(' L ')}" fill="none" stroke="${o.color}" stroke-width="2"/>`;
      const last = pts[pts.length - 1];
      pathsHtml += `<circle cx="${xToPx(last[0])}" cy="${yToPx(last[1])}" r="5" fill="${o.color}"/>`;
      const loss = L(last[0], last[1]);
      legendHtml += `<tspan x="460" dy="16" fill="${o.color}" font-weight="bold">${o.name}</tspan>` +
                    `<tspan x="580" text-anchor="end" fill="#374151">loss=${loss.toFixed(3)}</tspan>`;
    });
    svg.innerHTML = `
      ${drawContours()}
      <circle cx="${xToPx(3)}" cy="${yToPx(3)}" r="6" fill="none" stroke="#10b981" stroke-width="2"/>
      <text x="${xToPx(3) + 10}" y="${yToPx(3) + 4}" font-size="11" fill="#10b981">min</text>
      <circle cx="${xToPx(startX)}" cy="${yToPx(startY)}" r="4" fill="#6b7280"/>
      <text x="${xToPx(startX) - 10}" y="${yToPx(startY) - 6}" font-size="11" fill="#6b7280" text-anchor="end">start</text>
      ${pathsHtml}
      <text x="460" y="24" font-size="12">${legendHtml}</text>
    `;
    out.textContent = `Step ${stepCount} / ${maxSteps}. Lower loss = closer to minimum.`;
  }

  function run() {
    if (intervalId) { clearInterval(intervalId); intervalId = null; runBtn.textContent = 'Run'; return; }
    runBtn.textContent = 'Stop';
    intervalId = setInterval(() => {
      if (stepCount >= maxSteps) { clearInterval(intervalId); intervalId = null; runBtn.textContent = 'Run'; return; }
      opts.forEach(o => o.opt.step());
      stepCount++;
      render();
    }, 30);
  }

  lrEl.addEventListener('input', () => { lrOut.textContent = parseFloat(lrEl.value).toFixed(3); reset(); });
  runBtn.addEventListener('click', run);
  resetBtn.addEventListener('click', () => { if (intervalId) { clearInterval(intervalId); intervalId = null; runBtn.textContent = 'Run'; } reset(); });
  reset();
});
