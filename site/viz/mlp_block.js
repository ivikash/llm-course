// viz/mlp_block.js
// Animated: input vector → expand 4x → GELU → compress → output. Sliders for dim.

registerViz('mlp_block', function (container) {
  container.innerHTML = `
    <p class="viz-title">MLP block: expand, activate, compress</p>
    <p class="viz-sub">Each token's vector enters at C dims, gets lifted to 4C dims (room to compute), activated, and projected back to C.</p>
    <div class="viz-controls">
      <label>C (embed dim): <input id="mlp-c" type="range" min="32" max="1024" step="32" value="128"><span id="mlp-c-out">128</span></label>
      <label>Activation:
        <select id="mlp-act">
          <option value="gelu" selected>GELU (nanoGPT)</option>
          <option value="swiglu">SwiGLU (Llama, nanochat)</option>
          <option value="relu">ReLU (classical)</option>
        </select>
      </label>
      <button id="mlp-play">▶ Animate one forward</button>
    </div>
    <svg id="mlp-svg" viewBox="0 0 760 260" width="760" height="260" style="background:#fff;border:1px solid #e5e7eb;border-radius:4px;max-width:100%"></svg>
    <p class="viz-readout" id="mlp-out"></p>
  `;

  const svg = container.querySelector('#mlp-svg');
  let phase = -1;

  function render() {
    const C = +container.querySelector('#mlp-c').value;
    const act = container.querySelector('#mlp-act').value;
    container.querySelector('#mlp-c-out').textContent = C;
    const stages = act === 'swiglu'
      ? [
          { label: 'input', shape: `(B, T, ${C})`,     x: 40,  w: 50 },
          { label: 'W_fc',   shape: `(B, T, ${4*C})`,   x: 140, w: 150 },
          { label: 'W_gate', shape: `(B, T, ${4*C})`,   x: 140, w: 150, gate: true },
          { label: 'SiLU×',  shape: `(B, T, ${4*C})`,   x: 340, w: 150 },
          { label: 'W_proj', shape: `(B, T, ${C})`,    x: 540, w: 50 },
          { label: 'output', shape: `(B, T, ${C})`,    x: 660, w: 50 },
        ]
      : [
          { label: 'input', shape: `(B, T, ${C})`,     x: 40,  w: 50 },
          { label: 'c_fc',   shape: `(B, T, ${4*C})`,   x: 200, w: 150 },
          { label: act === 'gelu' ? 'GELU' : 'ReLU', shape: `(B, T, ${4*C})`, x: 400, w: 150 },
          { label: 'c_proj', shape: `(B, T, ${C})`,    x: 580, w: 50 },
          { label: 'output', shape: `(B, T, ${C})`,    x: 700, w: 50 },
        ];
    let html = '';
    for (let i = 0; i < stages.length; i++) {
      const s = stages[i];
      const active = phase === i;
      const done = phase > i || phase === -1;
      const fill = active ? '#fbbf24' : done ? '#dbeafe' : '#f3f4f6';
      const y = s.gate ? 140 : 50;
      html += `<rect x="${s.x}" y="${y}" width="${s.w}" height="80" fill="${fill}" stroke="#6b7280" rx="6"/>`;
      html += `<text x="${s.x + s.w/2}" y="${y + 30}" font-size="12" text-anchor="middle" fill="#111" font-weight="bold">${s.label}</text>`;
      html += `<text x="${s.x + s.w/2}" y="${y + 52}" font-size="10" text-anchor="middle" fill="#6b7280" font-family="ui-monospace,monospace">${s.shape}</text>`;
      if (i < stages.length - 1 && !s.gate) {
        const next = stages[i + 1];
        const arrowY = y + 40;
        html += `<line x1="${s.x + s.w + 4}" y1="${arrowY}" x2="${next.x - 4}" y2="${arrowY}" stroke="#6b7280" stroke-width="1.5" marker-end="url(#mlp-arr)"/>`;
      }
    }
    if (act === 'swiglu') {
      // draw gate merge arrow
      html += `<line x1="290" y1="180" x2="340" y2="130" stroke="#6b7280" stroke-width="1.5" marker-end="url(#mlp-arr)"/>`;
    }
    html = `<defs><marker id="mlp-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto"><path d="M0 0 L10 5 L0 10 z" fill="#6b7280"/></marker></defs>` + html;
    svg.innerHTML = html;

    // param counts
    let params;
    if (act === 'swiglu') params = C * 4*C + C * 4*C + 4*C * C;    // fc + gate + proj
    else params = C * 4*C + 4*C * C;
    container.querySelector('#mlp-out').innerHTML =
      `Params in this MLP: <b>${params.toLocaleString()}</b> ` +
      `(~${act === 'swiglu' ? '12C²' : '8C²'} per block). ` +
      `Attention has ~4C². Most of a transformer's weight is in its MLPs.`;
  }

  async function play() {
    const C = +container.querySelector('#mlp-c').value;
    const act = container.querySelector('#mlp-act').value;
    const total = act === 'swiglu' ? 6 : 5;
    for (let i = 0; i < total; i++) {
      phase = i;
      render();
      await new Promise(r => setTimeout(r, 450));
    }
    phase = -1;
    render();
  }

  container.querySelector('#mlp-c').addEventListener('input', render);
  container.querySelector('#mlp-act').addEventListener('change', render);
  container.querySelector('#mlp-play').addEventListener('click', play);
  render();
});
