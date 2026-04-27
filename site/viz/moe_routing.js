// viz/moe_routing.js
// Mixture of Experts: router picks top-k experts per token.

registerViz('moe_routing', function (container) {
  container.innerHTML = `
    <p class="viz-title">Mixture of Experts (MoE) routing</p>
    <p class="viz-sub">Each token gets routed to top-k of N experts. Only those experts run. Way more capacity, same compute per token.</p>
    <div class="viz-controls">
      <label>Experts: <input id="mr-n" type="range" min="4" max="16" value="8"><span id="mr-n-out">8</span></label>
      <label>Top-k: <input id="mr-k" type="range" min="1" max="4" value="2"><span id="mr-k-out">2</span></label>
      <button id="mr-step">New batch</button>
    </div>
    <svg id="mr-svg" viewBox="0 0 780 300" width="780" height="300" style="background:#fff;border:1px solid #e5e7eb;border-radius:4px;max-width:100%;margin-top:12px"></svg>
    <p class="viz-readout" id="mr-out"></p>
  `;
  const tokens = ['the', 'cat', 'sat', 'on', 'the', 'mat', 'quickly', 'today'];

  function render() {
    const N = +container.querySelector('#mr-n').value;
    const K = +container.querySelector('#mr-k').value;
    container.querySelector('#mr-n-out').textContent = N;
    container.querySelector('#mr-k-out').textContent = K;

    // Each token routes to K random experts (deterministic by token idx+N+step for fairness)
    const T = tokens.length;
    const assignments = tokens.map(() => {
      const scores = Array.from({ length: N }, () => Math.random());
      const ranked = scores.map((s, i) => [s, i]).sort((a, b) => b[0] - a[0]);
      return ranked.slice(0, K).map(r => r[1]);
    });

    const svg = container.querySelector('#mr-svg');
    let html = '';

    // tokens on left
    const tokY = (i) => 30 + i * 32;
    tokens.forEach((t, i) => {
      html += `<rect x="20" y="${tokY(i) - 12}" width="100" height="24" fill="#dbeafe" rx="4"/>`;
      html += `<text x="70" y="${tokY(i) + 4}" font-size="12" text-anchor="middle" fill="#1e3a8a" font-family="ui-monospace,monospace">${t}</text>`;
    });

    // router in middle
    html += `<rect x="160" y="120" width="120" height="40" fill="#fef3c7" stroke="#d97706" rx="6"/>`;
    html += `<text x="220" y="144" font-size="12" text-anchor="middle" fill="#78350f" font-weight="bold">router (top-${K})</text>`;

    // experts on right
    const expX = 520;
    const expH = 260 / N;
    const expUsed = new Set();
    assignments.forEach(a => a.forEach(e => expUsed.add(e)));
    for (let e = 0; e < N; e++) {
      const y = 20 + e * expH;
      const used = expUsed.has(e);
      html += `<rect x="${expX}" y="${y}" width="120" height="${expH - 4}" fill="${used ? '#bbf7d0' : '#f3f4f6'}" stroke="${used ? '#15803d' : '#d1d5db'}" rx="4"/>`;
      html += `<text x="${expX + 60}" y="${y + expH/2}" font-size="11" text-anchor="middle" fill="${used ? '#14532d' : '#9ca3af'}">expert ${e}</text>`;
    }

    // token -> router lines
    tokens.forEach((t, i) => {
      html += `<line x1="120" y1="${tokY(i)}" x2="160" y2="140" stroke="#9ca3af" stroke-width="0.5"/>`;
    });

    // router -> expert lines (colored per token)
    const colors = ['#3b82f6','#f59e0b','#10b981','#8b5cf6','#ec4899','#06b6d4','#f97316','#eab308'];
    tokens.forEach((t, i) => {
      assignments[i].forEach(e => {
        const ey = 20 + e * expH + expH / 2;
        html += `<path d="M 280 140 Q 400 140 ${expX} ${ey}" stroke="${colors[i]}" stroke-width="1.5" fill="none" opacity="0.7"/>`;
      });
    });

    svg.innerHTML = html;

    // stats
    const total = T * K;
    const unique = expUsed.size;
    const activeFrac = (unique / N * 100).toFixed(0);
    const paramEffBoost = N / K;
    container.querySelector('#mr-out').innerHTML =
      `${T} tokens × top-${K} = ${total} expert dispatches. ${unique}/${N} experts active this step (${activeFrac}%). ` +
      `<b>Parameter efficiency: ${paramEffBoost.toFixed(1)}× more params than dense for the same FLOPs.</b> ` +
      `Mixtral-8×7B has 47B total params but uses only ~13B per token.`;
  }
  container.querySelector('#mr-n').addEventListener('input', render);
  container.querySelector('#mr-k').addEventListener('input', render);
  container.querySelector('#mr-step').addEventListener('click', render);
  render();
});
