// viz/entropy_curve.js
// For a binary distribution (p, 1-p), show entropy H(p) = -p log p - (1-p) log(1-p).
// Max at p=0.5. Ties into cross-entropy + information theory intuition.
registerViz('entropy_curve', async function (container) {
  const d3 = await ensureD3();
  container.innerHTML = `
    <p class="viz-title">Entropy: how unpredictable is a distribution?</p>
    <p class="viz-sub">Binary entropy H(p)=-p·log₂p - (1-p)·log₂(1-p). Max at p=0.5 (maximum uncertainty). Zero at p=0 or 1.</p>
    <div class="viz-controls">
      <label>p: <input id="ec-p" type="range" min="0.001" max="0.999" step="0.001" value="0.5"><span id="ec-p-o">0.500</span></label>
    </div>
    <svg width="540" height="280"></svg>
    <p class="viz-readout" id="ec-out"></p>`;
  const svg = d3.select(container).select('svg');
  const W=540, H=280, M=30;
  const x = d3.scaleLinear().domain([0,1]).range([M, W-M]);
  const y = d3.scaleLinear().domain([0, 1.1]).range([H-M, M]);
  svg.append('g').attr('transform',`translate(0,${H-M})`).call(d3.axisBottom(x));
  svg.append('g').attr('transform',`translate(${M},0)`).call(d3.axisLeft(y));
  const H2 = p => (p<=0||p>=1 ? 0 : -(p*Math.log2(p) + (1-p)*Math.log2(1-p)));
  const pts = d3.range(0.001, 1, 0.005).map(p => [x(p), y(H2(p))]);
  svg.append('path').attr('d', d3.line()(pts)).attr('fill','none').attr('stroke','#2563eb').attr('stroke-width',2);
  const dot = svg.append('circle').attr('r',7).attr('fill','#dc2626');
  function draw() {
    const p = +container.querySelector('#ec-p').value;
    container.querySelector('#ec-p-o').textContent = p.toFixed(3);
    dot.attr('cx', x(p)).attr('cy', y(H2(p)));
    container.querySelector('#ec-out').textContent =
      `H(p=${p.toFixed(3)}) = ${H2(p).toFixed(3)} bits. Cross-entropy loss in LLMs is the expected number of bits needed to encode each token when using the model's probability distribution.`;
  }
  container.querySelector('#ec-p').addEventListener('input', draw);
  draw();
});
