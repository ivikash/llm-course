// viz/eigenvectors_2d.js
// Apply matrix M repeatedly to random unit vectors. They align with top eigenvector.
registerViz('eigenvectors_2d', async function (container) {
  const d3 = await ensureD3();
  container.innerHTML = `
    <p class="viz-title">Eigenvectors emerge under repeated application</p>
    <p class="viz-sub">Random unit vectors, each multiplied by M many times and renormalized, collapse onto the top eigenvector.</p>
    <div class="viz-controls">
      <button id="ev-step">Step ×1</button>
      <button id="ev-step10">Step ×10</button>
      <button id="ev-reset">Reset</button>
      <span id="ev-iter">iter=0</span>
    </div>
    <svg width="420" height="420"></svg>
    <p class="viz-readout" id="ev-out"></p>`;
  const svg = d3.select(container).select('svg');
  const W=420, H=420, ox=W/2, oy=H/2, R=140;
  // Fixed matrix with eigenvalues 2 and 0.5
  const M = [[1.6, 0.6],[0.4, 0.9]];
  let vecs = d3.range(20).map(() => {
    const th = Math.random()*2*Math.PI; return [Math.cos(th), Math.sin(th)];
  });
  let iter = 0;
  function draw() {
    svg.selectAll('*').remove();
    svg.append('circle').attr('cx',ox).attr('cy',oy).attr('r',R).attr('fill','none').attr('stroke','#e5e7eb');
    for (const v of vecs) {
      const n = Math.hypot(v[0],v[1]);
      const [x,y] = [v[0]/n, v[1]/n];
      svg.append('line').attr('x1',ox).attr('y1',oy).attr('x2',ox+x*R).attr('y2',oy-y*R)
        .attr('stroke','#2563eb').attr('stroke-width',1.5).attr('opacity',0.6);
    }
    container.querySelector('#ev-iter').textContent = `iter=${iter}`;
    // Compute dominant direction estimate
    let sx=0, sy=0; for (const v of vecs) { const n=Math.hypot(v[0],v[1]); const s=Math.sign(v[0]);
      sx += s*v[0]/n; sy += s*v[1]/n; }
    sx/=vecs.length; sy/=vecs.length;
    container.querySelector('#ev-out').textContent =
      `M = [[1.6,0.6],[0.4,0.9]]. Eigenvalues ≈ 2.0 and 0.5. After many iters, all vectors align with the top eigenvector. Estimated dir: (${sx.toFixed(2)},${sy.toFixed(2)}).`;
  }
  function step(n=1) {
    for (let k=0;k<n;k++) {
      vecs = vecs.map(([x,y]) => {
        const nx = M[0][0]*x + M[0][1]*y, ny = M[1][0]*x + M[1][1]*y;
        const nm = Math.hypot(nx, ny) || 1;
        return [nx/nm, ny/nm];
      });
      iter++;
    }
    draw();
  }
  container.querySelector('#ev-step').onclick = () => step(1);
  container.querySelector('#ev-step10').onclick = () => step(10);
  container.querySelector('#ev-reset').onclick = () => {
    vecs = d3.range(20).map(() => { const th=Math.random()*2*Math.PI; return [Math.cos(th),Math.sin(th)]; });
    iter = 0; draw();
  };
  draw();
});
