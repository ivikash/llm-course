// viz/derivative_as_slope.js
// Drag a point along a curve; see the tangent line and df/dx numerically.
// The single most useful calculus viz for DL.
registerViz('derivative_as_slope', async function (container) {
  const d3 = await ensureD3();
  container.innerHTML = `
    <p class="viz-title">Derivative = local slope</p>
    <p class="viz-sub">Drag the red point. The tangent shows df/dx at that x.</p>
    <div class="viz-controls">
      <label>f(x)=
        <select id="ds-fn">
          <option value="sq">x²</option><option value="cube">x³</option>
          <option value="sin">sin(x)</option><option value="exp">eˣ</option>
        </select>
      </label>
    </div>
    <svg width="560" height="340"></svg>
    <p class="viz-readout" id="ds-out"></p>`;
  const svg = d3.select(container).select('svg');
  const W = 560, H = 340, M = 30;
  const x = d3.scaleLinear().domain([-3, 3]).range([M, W - M]);
  const y = d3.scaleLinear().domain([-4, 8]).range([H - M, M]);
  svg.append('g').attr('transform', `translate(0,${H/2})`).call(d3.axisBottom(x));
  svg.append('g').attr('transform', `translate(${W/2},0)`).call(d3.axisLeft(y));
  const fns = {
    sq:   { f: v => v*v,     df: v => 2*v },
    cube: { f: v => v*v*v,   df: v => 3*v*v },
    sin:  { f: Math.sin,     df: Math.cos },
    exp:  { f: Math.exp,     df: Math.exp },
  };
  let fn = fns.sq, xp = 1;
  const path = svg.append('path').attr('fill', 'none').attr('stroke', '#2563eb').attr('stroke-width', 2);
  const tan  = svg.append('line').attr('stroke', '#dc2626').attr('stroke-width', 2).attr('stroke-dasharray', '4 4');
  const dot  = svg.append('circle').attr('r', 7).attr('fill', '#dc2626').style('cursor', 'grab');
  const out  = container.querySelector('#ds-out');
  function draw() {
    const pts = d3.range(-3, 3.02, 0.02).map(v => [x(v), y(fn.f(v))]);
    path.attr('d', d3.line()(pts));
    const slope = fn.df(xp);
    const px = x(xp), py = y(fn.f(xp));
    const dx = 60;
    tan.attr('x1', px - dx).attr('y1', py + (slope * (x.invert(px-dx) - xp)) * (y(0)-y(1))/(0-1) * 0)
       .attr('x2', px + dx).attr('y2', py - (slope * (x.invert(px+dx) - xp)) * (y(0)-y(1))/(0-1) * 0);
    // Tangent: use data-space line
    const x1 = xp - 1, x2 = xp + 1;
    tan.attr('x1', x(x1)).attr('y1', y(fn.f(xp) + slope*(x1-xp)))
       .attr('x2', x(x2)).attr('y2', y(fn.f(xp) + slope*(x2-xp)));
    dot.attr('cx', px).attr('cy', py);
    out.textContent = `x=${xp.toFixed(2)}  f(x)=${fn.f(xp).toFixed(3)}  df/dx=${slope.toFixed(3)}`;
  }
  dot.call(d3.drag().on('drag', ev => { xp = Math.max(-3, Math.min(3, x.invert(ev.x))); draw(); }));
  container.querySelector('#ds-fn').addEventListener('change', e => { fn = fns[e.target.value]; draw(); });
  draw();
});
