// viz/chain_rule_flow.js
// y = g(f(x)). Shows δx → δf → δy scaling through f' then g'. Backprop intuition.
registerViz('chain_rule_flow', async function (container) {
  const d3 = await ensureD3();
  container.innerHTML = `
    <p class="viz-title">Chain rule = gradients multiply through a pipeline</p>
    <p class="viz-sub">y = g(f(x)). A nudge δx becomes f'(x)·δx after f, then g'(f(x))·f'(x)·δx after g.</p>
    <div class="viz-controls">
      <label>x: <input id="cr-x" type="range" min="-2" max="2" step="0.05" value="1"><span id="cr-x-out">1</span></label>
      <label>δx: <input id="cr-dx" type="range" min="0.01" max="0.5" step="0.01" value="0.2"><span id="cr-dx-out">0.2</span></label>
    </div>
    <svg width="560" height="260"></svg>
    <p class="viz-readout" id="cr-out"></p>`;
  const svg = d3.select(container).select('svg');
  const f = v => v*v, fp = v => 2*v;        // f(x)=x²
  const g = v => Math.sin(v), gp = v => Math.cos(v); // g(u)=sin(u)
  function draw() {
    const xv = +container.querySelector('#cr-x').value;
    const dx = +container.querySelector('#cr-dx').value;
    container.querySelector('#cr-x-out').textContent = xv.toFixed(2);
    container.querySelector('#cr-dx-out').textContent = dx.toFixed(2);
    const fxv = f(xv), df = fp(xv)*dx;
    const gfv = g(fxv), dg = gp(fxv)*df;
    svg.selectAll('*').remove();
    const nodes = [
      { x: 60,  y: 130, label: 'x',       v: xv.toFixed(2),  d: dx.toFixed(2) },
      { x: 240, y: 130, label: 'f(x)=x²', v: fxv.toFixed(2), d: df.toFixed(3) },
      { x: 440, y: 130, label: 'g(u)=sin(u)', v: gfv.toFixed(2), d: dg.toFixed(3) },
    ];
    svg.selectAll('line.edge').data([[nodes[0], nodes[1], `×f'(x)=${fp(xv).toFixed(2)}`],
                                     [nodes[1], nodes[2], `×g'(f)=${gp(fxv).toFixed(2)}`]])
      .enter().append('g').each(function(d) {
        const g_ = d3.select(this);
        g_.append('line').attr('x1', d[0].x+50).attr('y1', d[0].y).attr('x2', d[1].x-50).attr('y2', d[1].y)
          .attr('stroke', '#6b7280').attr('marker-end', 'url(#arr)').attr('stroke-width', 2);
        g_.append('text').attr('x', (d[0].x+d[1].x)/2).attr('y', d[0].y-10)
          .attr('text-anchor','middle').attr('font-size', 12).attr('fill', '#374151').text(d[2]);
      });
    svg.append('defs').append('marker').attr('id','arr').attr('markerWidth',8).attr('markerHeight',8)
      .attr('refX',6).attr('refY',4).attr('orient','auto')
      .append('path').attr('d','M0,0 L8,4 L0,8').attr('fill','#6b7280');
    svg.selectAll('g.node').data(nodes).enter().append('g').each(function(d) {
      const g_ = d3.select(this);
      g_.append('circle').attr('cx',d.x).attr('cy',d.y).attr('r',44).attr('fill','#dbeafe').attr('stroke','#2563eb');
      g_.append('text').attr('x',d.x).attr('y',d.y-6).attr('text-anchor','middle').attr('font-size',12).attr('font-weight',600).text(d.label);
      g_.append('text').attr('x',d.x).attr('y',d.y+10).attr('text-anchor','middle').attr('font-size',11).text('val='+d.v);
      g_.append('text').attr('x',d.x).attr('y',d.y+26).attr('text-anchor','middle').attr('font-size',11).attr('fill','#dc2626').text('δ='+d.d);
    });
    container.querySelector('#cr-out').textContent =
      `dy/dx = g'(f(x)) · f'(x) = ${gp(fxv).toFixed(3)} × ${fp(xv).toFixed(3)} = ${(gp(fxv)*fp(xv)).toFixed(3)}`;
  }
  container.querySelectorAll('input').forEach(el => el.addEventListener('input', draw));
  draw();
});
