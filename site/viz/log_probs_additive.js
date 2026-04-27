// viz/log_probs_additive.js
// Three probability bars. Show P = p1*p2*p3 vs logP = logp1+logp2+logp3.
registerViz('log_probs_additive', async function (container) {
  container.innerHTML = `
    <p class="viz-title">Why we sum log-probs instead of multiplying probs</p>
    <p class="viz-sub">Probabilities multiply (underflow fast). Logs add (stable). Language-model loss sums log-probs.</p>
    <div class="viz-controls">
      <label>p₁: <input id="lp-1" type="range" min="0.001" max="1" step="0.001" value="0.7"><span id="lp-1-o">0.7</span></label>
      <label>p₂: <input id="lp-2" type="range" min="0.001" max="1" step="0.001" value="0.4"><span id="lp-2-o">0.4</span></label>
      <label>p₃: <input id="lp-3" type="range" min="0.001" max="1" step="0.001" value="0.3"><span id="lp-3-o">0.3</span></label>
    </div>
    <p class="viz-readout" id="lp-out"></p>`;
  const out = container.querySelector('#lp-out');
  function draw() {
    const p1 = +container.querySelector('#lp-1').value;
    const p2 = +container.querySelector('#lp-2').value;
    const p3 = +container.querySelector('#lp-3').value;
    container.querySelector('#lp-1-o').textContent = p1.toFixed(3);
    container.querySelector('#lp-2-o').textContent = p2.toFixed(3);
    container.querySelector('#lp-3-o').textContent = p3.toFixed(3);
    const P = p1*p2*p3, lP = Math.log(p1)+Math.log(p2)+Math.log(p3);
    out.textContent =
      `P = p₁·p₂·p₃ = ${p1.toFixed(3)} × ${p2.toFixed(3)} × ${p3.toFixed(3)} = ${P.toExponential(3)}\n` +
      `log P = log p₁ + log p₂ + log p₃ = ${Math.log(p1).toFixed(3)} + ${Math.log(p2).toFixed(3)} + ${Math.log(p3).toFixed(3)} = ${lP.toFixed(3)}\n` +
      `\nIn an LLM, a sequence probability is Π P(token_t | ...). With 1000 tokens at p=0.1 each, product = 1e-1000 (underflow).\n` +
      `Sum of logs = -2302.6 (perfectly fine number). This is why cross-entropy is a *sum*, not a product.`;
  }
  container.querySelectorAll('input').forEach(el => el.addEventListener('input', draw));
  draw();
});
