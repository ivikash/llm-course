// viz/kl_divergence.js
// Two discrete distributions P and Q as bars. Slide Q; see KL(P||Q) and asymmetry.
registerViz('kl_divergence', async function (container) {
  const d3 = await ensureD3();
  const K = 6;
  container.innerHTML = `
    <p class="viz-title">KL divergence: how far Q is from P</p>
    <p class="viz-sub">Drag Q's bars. KL(P‖Q)=Σ P log(P/Q). Critical for RLHF / GRPO KL penalty.</p>
    <svg width="560" height="300"></svg>
    <p class="viz-readout" id="kl-out"></p>`;
  const svg = d3.select(container).select('svg');
  const W=560, H=300, M=40, bw = (W-2*M)/K/2.2;
  const xs = d3.scaleBand().domain(d3.range(K)).range([M, W-M]).padding(0.3);
  const ys = d3.scaleLinear().domain([0,1]).range([H-M, M]);
  svg.append('g').attr('transform',`translate(0,${H-M})`).call(d3.axisBottom(xs));
  svg.append('g').attr('transform',`translate(${M},0)`).call(d3.axisLeft(ys));
  let P = [0.05,0.1,0.3,0.3,0.2,0.05];
  let Q = [0.17,0.17,0.16,0.17,0.17,0.16];
  function norm(a) { const s = a.reduce((x,y)=>x+y,0); return a.map(v=>v/s); }
  function kl(p, q) { let s = 0; for (let i=0;i<p.length;i++) if (p[i]>0 && q[i]>0) s += p[i]*Math.log(p[i]/q[i]); return s; }
  const out = container.querySelector('#kl-out');
  function draw() {
    svg.selectAll('g.bars').remove();
    const g = svg.append('g').attr('class','bars');
    P = norm(P); Q = norm(Q);
    for (let i=0;i<K;i++) {
      g.append('rect').attr('x', xs(i)).attr('width', bw).attr('y', ys(P[i])).attr('height', ys(0)-ys(P[i]))
        .attr('fill','#2563eb').attr('opacity',0.8);
      g.append('rect').attr('x', xs(i)+bw+2).attr('width', bw).attr('y', ys(Q[i])).attr('height', ys(0)-ys(Q[i]))
        .attr('fill','#f59e0b').style('cursor','ns-resize')
        .call(d3.drag().on('drag', ev => {
          const nq = Math.max(0.01, Math.min(0.95, ys.invert(ev.y)));
          Q[i] = nq; draw();
        }));
    }
    const klPQ = kl(P, Q), klQP = kl(Q, P);
    out.textContent = `KL(P‖Q) = ${klPQ.toFixed(3)} nats   KL(Q‖P) = ${klQP.toFixed(3)} nats   (asymmetric!)`;
    svg.selectAll('text.legend').remove();
    svg.append('text').attr('class','legend').attr('x',M).attr('y',22).attr('font-size',12).attr('fill','#2563eb').text('■ P (fixed)');
    svg.append('text').attr('class','legend').attr('x',M+80).attr('y',22).attr('font-size',12).attr('fill','#f59e0b').text('■ Q (drag bars)');
  }
  draw();
});
