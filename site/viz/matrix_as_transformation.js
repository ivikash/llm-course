// viz/matrix_as_transformation.js
// 2D grid mapped by a 2x2 matrix. Sliders for the 4 entries. Shows rotation, scale, shear.
registerViz('matrix_as_transformation', async function (container) {
  const d3 = await ensureD3();
  container.innerHTML = `
    <p class="viz-title">A matrix is a transformation of space</p>
    <p class="viz-sub">The 2×2 matrix [[a,b],[c,d]] sends the blue grid to the red grid. Embeddings live in spaces like this.</p>
    <div class="viz-controls">
      <label>a: <input id="mt-a" type="range" min="-2" max="2" step="0.05" value="1"><span id="mt-a-o">1.00</span></label>
      <label>b: <input id="mt-b" type="range" min="-2" max="2" step="0.05" value="0.5"><span id="mt-b-o">0.50</span></label>
      <label>c: <input id="mt-c" type="range" min="-2" max="2" step="0.05" value="0"><span id="mt-c-o">0.00</span></label>
      <label>d: <input id="mt-d" type="range" min="-2" max="2" step="0.05" value="1"><span id="mt-d-o">1.00</span></label>
    </div>
    <svg width="480" height="380"></svg>
    <p class="viz-readout" id="mt-out"></p>`;
  const svg = d3.select(container).select('svg');
  const W=480, H=380;
  const s = 40, ox=W/2, oy=H/2;
  function draw() {
    const a = +container.querySelector('#mt-a').value, b = +container.querySelector('#mt-b').value;
    const c = +container.querySelector('#mt-c').value, d = +container.querySelector('#mt-d').value;
    ['a','b','c','d'].forEach(k => container.querySelector('#mt-'+k+'-o').textContent = (+container.querySelector('#mt-'+k).value).toFixed(2));
    svg.selectAll('*').remove();
    // Blue grid (original)
    for (let i=-3;i<=3;i++) {
      svg.append('line').attr('x1',ox-3*s).attr('x2',ox+3*s).attr('y1',oy+i*s).attr('y2',oy+i*s).attr('stroke','#dbeafe').attr('stroke-width',1);
      svg.append('line').attr('y1',oy-3*s).attr('y2',oy+3*s).attr('x1',ox+i*s).attr('x2',ox+i*s).attr('stroke','#dbeafe').attr('stroke-width',1);
    }
    // Red transformed grid
    function tx(x,y) { return [ox + (a*x + b*y)*s, oy - (c*x + d*y)*s]; }
    for (let i=-3;i<=3;i++) {
      for (let t=-3;t<3;t+=0.2) {
        const [x1,y1] = tx(t, i), [x2,y2] = tx(t+0.2, i);
        svg.append('line').attr('x1',x1).attr('y1',y1).attr('x2',x2).attr('y2',y2).attr('stroke','#fecaca').attr('stroke-width',1);
        const [u1,v1] = tx(i, t), [u2,v2] = tx(i, t+0.2);
        svg.append('line').attr('x1',u1).attr('y1',v1).attr('x2',u2).attr('y2',v2).attr('stroke','#fecaca').attr('stroke-width',1);
      }
    }
    // Basis vectors
    const [e1x,e1y] = tx(1,0), [e2x,e2y] = tx(0,1);
    svg.append('line').attr('x1',ox).attr('y1',oy).attr('x2',e1x).attr('y2',e1y).attr('stroke','#dc2626').attr('stroke-width',3);
    svg.append('line').attr('x1',ox).attr('y1',oy).attr('x2',e2x).attr('y2',e2y).attr('stroke','#16a34a').attr('stroke-width',3);
    const det = a*d - b*c;
    container.querySelector('#mt-out').textContent =
      `det = ad-bc = ${det.toFixed(2)} (area scaling factor; negative = flip). ê₁→(${a.toFixed(2)},${c.toFixed(2)})  ê₂→(${b.toFixed(2)},${d.toFixed(2)})`;
  }
  container.querySelectorAll('input').forEach(el => el.addEventListener('input', draw));
  draw();
});
