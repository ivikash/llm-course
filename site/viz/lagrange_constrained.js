// viz/lagrange_constrained.js
// Minimize f(x,y)=x²+y² subject to x+y=c. Shows gradient of f and constraint line.
registerViz('lagrange_constrained', async function (container) {
  const d3 = await ensureD3();
  container.innerHTML = `
    <p class="viz-title">Lagrange multipliers: constrained optimization</p>
    <p class="viz-sub">Minimize f(x,y)=x²+y² on the line x+y=c. The optimum is where ∇f is parallel to the constraint normal.</p>
    <div class="viz-controls">
      <label>c: <input id="lg-c" type="range" min="-2" max="2" step="0.1" value="1"><span id="lg-c-o">1.0</span></label>
    </div>
    <svg width="440" height="380"></svg>
    <p class="viz-readout" id="lg-out"></p>`;
  const svg = d3.select(container).select('svg');
  const W=440, H=380, M=30;
  const xs = d3.scaleLinear().domain([-3,3]).range([M,W-M]);
  const ys = d3.scaleLinear().domain([-3,3]).range([H-M,M]);
  function draw() {
    const c = +container.querySelector('#lg-c').value;
    container.querySelector('#lg-c-o').textContent = c.toFixed(1);
    svg.selectAll('*').remove();
    // Contours of f=r²
    [0.5,1,2,3.5,5].forEach(v => svg.append('circle')
      .attr('cx', xs(0)).attr('cy', ys(0)).attr('rx', xs(Math.sqrt(v))-xs(0)).attr('ry', ys(0)-ys(Math.sqrt(v)))
      .attr('r', xs(Math.sqrt(v))-xs(0))
      .attr('fill','none').attr('stroke','#93c5fd'));
    svg.append('g').attr('transform',`translate(0,${ys(0)})`).call(d3.axisBottom(xs));
    svg.append('g').attr('transform',`translate(${xs(0)},0)`).call(d3.axisLeft(ys));
    // Constraint line: x+y = c
    svg.append('line').attr('x1', xs(-3)).attr('y1', ys(c+3)).attr('x2', xs(3)).attr('y2', ys(c-3))
      .attr('stroke','#dc2626').attr('stroke-width',2);
    // Optimum: x=y=c/2
    const ox = c/2, oy = c/2;
    svg.append('circle').attr('cx', xs(ox)).attr('cy', ys(oy)).attr('r',8).attr('fill','#16a34a');
    container.querySelector('#lg-out').textContent =
      `min is at (c/2, c/2) = (${ox.toFixed(2)}, ${oy.toFixed(2)}). ∇f=(2x,2y) parallel to constraint normal (1,1). Lagrange multiplier λ = c.`;
  }
  container.querySelectorAll('input').forEach(el => el.addEventListener('input', draw));
  draw();
});
