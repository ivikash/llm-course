// viz/laplace_demo.js
// Impulse response of a damped oscillator. Slider: damping coefficient.
// Illustrates how a Laplace-domain pole becomes exponential decay × oscillation in time.
registerViz('laplace_demo', async function (container) {
  const d3 = await ensureD3();
  container.innerHTML = `
    <p class="viz-title">Laplace: poles ↔ time-domain behavior</p>
    <p class="viz-sub">Impulse response of ẍ + 2ζωẋ + ω²x = 0. ζ sets damping; ω sets frequency.</p>
    <div class="viz-controls">
      <label>ω (freq): <input id="lp-w" type="range" min="0.5" max="6" step="0.1" value="3"><span id="lp-w-o">3.0</span></label>
      <label>ζ (damp): <input id="lp-z" type="range" min="0" max="1.2" step="0.02" value="0.1"><span id="lp-z-o">0.10</span></label>
    </div>
    <svg width="540" height="260"></svg>
    <p class="viz-readout" id="lp-out"></p>`;
  const svg = d3.select(container).select('svg');
  const W=540, H=260, M=30;
  function draw() {
    const w = +container.querySelector('#lp-w').value, z = +container.querySelector('#lp-z').value;
    container.querySelector('#lp-w-o').textContent = w.toFixed(1);
    container.querySelector('#lp-z-o').textContent = z.toFixed(2);
    svg.selectAll('*').remove();
    const x = d3.scaleLinear().domain([0, 6]).range([M, W-M]);
    const y = d3.scaleLinear().domain([-1.2, 1.2]).range([H-M, M]);
    svg.append('g').attr('transform',`translate(0,${y(0)})`).call(d3.axisBottom(x));
    const pts = d3.range(0, 6.02, 0.02).map(t => {
      const wd = w*Math.sqrt(Math.max(0, 1 - z*z));
      const v = z < 1
        ? Math.exp(-z*w*t) * Math.cos(wd*t)
        : Math.exp(-w*t)*(1 + w*t); // critically-/overdamped approx
      return [x(t), y(v)];
    });
    svg.append('path').attr('d', d3.line()(pts)).attr('fill','none').attr('stroke','#2563eb').attr('stroke-width',2);
    container.querySelector('#lp-out').textContent =
      z < 1 ? `Underdamped: poles at s = -ζω ± jω√(1-ζ²) = ${(-z*w).toFixed(2)} ± j${(w*Math.sqrt(1-z*z)).toFixed(2)}` :
      z === 1 ? `Critically damped: double pole at s = -ω = ${-w.toFixed(2)}` :
      `Overdamped: two real poles at s = -ζω ± ω√(ζ²-1)`;
  }
  container.querySelectorAll('input').forEach(el => el.addEventListener('input', draw));
  draw();
});
