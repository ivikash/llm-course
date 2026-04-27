// viz/integral_area.js
// Numerical integration of f(x) via Riemann rectangles. Slider: number of rectangles.
registerViz('integral_area', async function (container) {
  const d3 = await ensureD3();
  container.innerHTML = `
    <p class="viz-title">Integral = area under curve (Riemann sum)</p>
    <p class="viz-sub">More rectangles → better area estimate → the integral.</p>
    <div class="viz-controls">
      <label>f(x)=<select id="in-fn"><option value="sq">x²</option><option value="sin">sin(x)</option><option value="bell">e^(-x²)</option></select></label>
      <label>a: <input id="in-a" type="range" min="-3" max="0" step="0.1" value="0"><span id="in-a-o">0</span></label>
      <label>b: <input id="in-b" type="range" min="0.1" max="3" step="0.1" value="2"><span id="in-b-o">2</span></label>
      <label>n: <input id="in-n" type="range" min="2" max="100" step="1" value="10"><span id="in-n-o">10</span></label>
    </div>
    <svg width="540" height="320"></svg>
    <p class="viz-readout" id="in-out"></p>`;
  const svg = d3.select(container).select('svg');
  const W=540, H=320, M=30;
  const fns = { sq: v=>v*v, sin: Math.sin, bell: v=>Math.exp(-v*v) };
  function draw() {
    const f = fns[container.querySelector('#in-fn').value];
    const a = +container.querySelector('#in-a').value, b = +container.querySelector('#in-b').value;
    const n = +container.querySelector('#in-n').value;
    ['a','b','n'].forEach(k => container.querySelector('#in-'+k+'-o').textContent = container.querySelector('#in-'+k).value);
    svg.selectAll('*').remove();
    const x = d3.scaleLinear().domain([-3.2, 3.2]).range([M, W-M]);
    const xs = d3.range(-3.2, 3.22, 0.02), ysv = xs.map(f);
    const ymin = Math.min(0, d3.min(ysv)) - 0.2;
    const ymax = Math.max(1, d3.max(ysv)) + 0.2;
    const y = d3.scaleLinear().domain([ymin, ymax]).range([H-M, M]);
    svg.append('g').attr('transform',`translate(0,${y(0)})`).call(d3.axisBottom(x));
    svg.append('path').attr('d', d3.line()(xs.map(v => [x(v), y(f(v))])))
      .attr('fill','none').attr('stroke','#2563eb').attr('stroke-width',2);
    const dx = (b-a)/n; let s = 0;
    for (let i=0;i<n;i++) {
      const xi = a + (i+0.5)*dx, fi = f(xi);
      s += fi*dx;
      svg.append('rect').attr('x', x(a+i*dx)).attr('y', y(Math.max(0,fi)))
        .attr('width', x(a+dx)-x(a)).attr('height', Math.abs(y(0)-y(fi)))
        .attr('fill','#f59e0b').attr('opacity', 0.5).attr('stroke','#d97706');
    }
    container.querySelector('#in-out').textContent =
      `∫ from ${a.toFixed(2)} to ${b.toFixed(2)} of f(x)dx ≈ ${s.toFixed(4)}  (n=${n} rectangles)`;
  }
  container.querySelectorAll('input,select').forEach(el => el.addEventListener('input', draw));
  draw();
});
