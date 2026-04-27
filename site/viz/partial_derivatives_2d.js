// viz/partial_derivatives_2d.js
// Loss surface L(w1,w2); drag point; show ∂L/∂w1 and ∂L/∂w2 as orthogonal slopes.
registerViz('partial_derivatives_2d', async function (container) {
  const d3 = await ensureD3();
  container.innerHTML = `
    <p class="viz-title">Partials: gradient is a vector of slopes</p>
    <p class="viz-sub">Loss L(w₁,w₂)=w₁²+2w₂². Contours + drag point. Arrow = ∇L (negative descent dir).</p>
    <svg width="540" height="400"></svg>
    <p class="viz-readout" id="pd-out"></p>`;
  const svg = d3.select(container).select('svg');
  const W=540, H=400, M=30;
  const xs = d3.scaleLinear().domain([-3,3]).range([M, W-M]);
  const ys = d3.scaleLinear().domain([-3,3]).range([H-M, M]);
  svg.append('g').attr('transform',`translate(0,${H/2})`).call(d3.axisBottom(xs));
  svg.append('g').attr('transform',`translate(${W/2},0)`).call(d3.axisLeft(ys));
  const L = (a,b) => a*a + 2*b*b;
  // Contours
  const cx = d3.range(-3,3.05,0.15), cy = d3.range(-3,3.05,0.15);
  const vals = cx.flatMap(a => cy.map(b => L(a,b)));
  const levels = [0.5,2,5,10,18];
  const contours = d3.contours().size([cx.length, cy.length]).thresholds(levels)(vals);
  svg.append('g').selectAll('path').data(contours).enter().append('path')
    .attr('d', d3.geoPath(d3.geoIdentity().scale((W-2*M)/cx.length).translate([M, M])))
    .attr('fill','none').attr('stroke','#93c5fd').attr('stroke-width',1);
  let p = {a: 2, b: 1.5};
  const arrow = svg.append('line').attr('stroke','#dc2626').attr('stroke-width',3).attr('marker-end','url(#pd-arr)');
  svg.append('defs').append('marker').attr('id','pd-arr').attr('markerWidth',10).attr('markerHeight',10)
    .attr('refX',8).attr('refY',5).attr('orient','auto')
    .append('path').attr('d','M0,0 L10,5 L0,10').attr('fill','#dc2626');
  const dot = svg.append('circle').attr('r',8).attr('fill','#dc2626').style('cursor','grab');
  const out = container.querySelector('#pd-out');
  function draw() {
    const px = xs(p.a), py = ys(p.b);
    const gx = 2*p.a, gy = 4*p.b;   // ∂L/∂w1, ∂L/∂w2
    const end = { x: xs(p.a - gx*0.15), y: ys(p.b - gy*0.15) }; // step opposite of gradient
    dot.attr('cx', px).attr('cy', py);
    arrow.attr('x1', px).attr('y1', py).attr('x2', end.x).attr('y2', end.y);
    out.textContent = `w=(${p.a.toFixed(2)}, ${p.b.toFixed(2)})  L=${L(p.a,p.b).toFixed(2)}  ∂L/∂w₁=${gx.toFixed(2)}  ∂L/∂w₂=${gy.toFixed(2)}  ∇L=(${gx.toFixed(2)},${gy.toFixed(2)})`;
  }
  dot.call(d3.drag().on('drag', ev => {
    p.a = Math.max(-3, Math.min(3, xs.invert(ev.x)));
    p.b = Math.max(-3, Math.min(3, ys.invert(ev.y)));
    draw();
  }));
  draw();
});
