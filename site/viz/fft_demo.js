// viz/fft_demo.js
// Compose a signal from sines; compute DFT magnitudes; show spectrum.
registerViz('fft_demo', async function (container) {
  const d3 = await ensureD3();
  container.innerHTML = `
    <p class="viz-title">FFT: a signal as a sum of sine waves</p>
    <p class="viz-sub">Adjust amplitudes of three frequencies. Top = time domain. Bottom = magnitudes from DFT.</p>
    <div class="viz-controls">
      <label>A(3 Hz): <input id="ff-a1" type="range" min="0" max="1" step="0.05" value="0.8"><span id="ff-a1-o">0.80</span></label>
      <label>A(7 Hz): <input id="ff-a2" type="range" min="0" max="1" step="0.05" value="0.3"><span id="ff-a2-o">0.30</span></label>
      <label>A(15 Hz): <input id="ff-a3" type="range" min="0" max="1" step="0.05" value="0.1"><span id="ff-a3-o">0.10</span></label>
    </div>
    <svg width="540" height="300"></svg>`;
  const svg = d3.select(container).select('svg');
  const W=540, H=300, M=30, N=128;
  function dft(x) {
    const mag = new Array(N/2).fill(0);
    for (let k=0; k<N/2; k++) {
      let re=0, im=0;
      for (let n=0; n<N; n++) {
        const th = -2*Math.PI*k*n/N;
        re += x[n]*Math.cos(th); im += x[n]*Math.sin(th);
      }
      mag[k] = Math.hypot(re, im)/N;
    }
    return mag;
  }
  function draw() {
    const a1 = +container.querySelector('#ff-a1').value;
    const a2 = +container.querySelector('#ff-a2').value;
    const a3 = +container.querySelector('#ff-a3').value;
    container.querySelector('#ff-a1-o').textContent = a1.toFixed(2);
    container.querySelector('#ff-a2-o').textContent = a2.toFixed(2);
    container.querySelector('#ff-a3-o').textContent = a3.toFixed(2);
    const x = d3.range(N).map(n => {
      const t = n/N;
      return a1*Math.sin(2*Math.PI*3*t) + a2*Math.sin(2*Math.PI*7*t) + a3*Math.sin(2*Math.PI*15*t);
    });
    svg.selectAll('*').remove();
    // time domain
    const tx = d3.scaleLinear().domain([0, N-1]).range([M, W-M]);
    const ty = d3.scaleLinear().domain([-1.5, 1.5]).range([130, M]);
    svg.append('path').attr('d', d3.line()((x.map((v,i)=>[tx(i), ty(v)])))).attr('fill','none').attr('stroke','#2563eb');
    svg.append('text').attr('x',M).attr('y',M-5).attr('font-size',11).attr('fill','#6b7280').text('time domain');
    // spectrum
    const mag = dft(x);
    const fx = d3.scaleBand().domain(d3.range(mag.length)).range([M, W-M]).padding(0.2);
    const fy = d3.scaleLinear().domain([0, 0.6]).range([H-M, 170]);
    svg.selectAll('rect').data(mag).enter().append('rect')
      .attr('x', (_,i)=>fx(i)).attr('width', fx.bandwidth())
      .attr('y', v=>fy(v)).attr('height', v=>H-M-fy(v)).attr('fill','#f59e0b');
    svg.append('text').attr('x',M).attr('y',165).attr('font-size',11).attr('fill','#6b7280').text('|DFT(x)| vs frequency bin');
  }
  container.querySelectorAll('input').forEach(el => el.addEventListener('input', draw));
  draw();
});
