// viz/transformer_block_flow.js
// Animated data flow through one full transformer block.
// Shows tensor shapes at each stage.

registerViz('transformer_block_flow', function (container) {
  container.innerHTML = `
    <p class="viz-title">Transformer block: data flow</p>
    <p class="viz-sub">Watch a tensor flow through one full block. Each arrow is a real tensor op. Shape stays the same end to end.</p>
    <div class="viz-controls">
      <label>B (batch): <input id="tb-b" type="range" min="1" max="32" value="4"><span id="tb-b-out">4</span></label>
      <label>T (seq): <input id="tb-t" type="range" min="8" max="2048" step="8" value="128"><span id="tb-t-out">128</span></label>
      <label>C (embed dim): <input id="tb-c" type="range" min="64" max="1024" step="64" value="768"><span id="tb-c-out">768</span></label>
      <label>heads: <input id="tb-h" type="range" min="1" max="16" value="12"><span id="tb-h-out">12</span></label>
      <button id="tb-play">▶ Play flow</button>
    </div>
    <svg id="tb-svg" viewBox="0 0 800 520" width="800" height="520" style="background:#fff;border:1px solid #e5e7eb;border-radius:4px;max-width:100%"></svg>
    <p class="viz-readout" id="tb-out"></p>
  `;

  const svg = container.querySelector('#tb-svg');
  const bEl = container.querySelector('#tb-b');
  const tEl = container.querySelector('#tb-t');
  const cEl = container.querySelector('#tb-c');
  const hEl = container.querySelector('#tb-h');
  const out = container.querySelector('#tb-out');
  let highlight = -1;

  function stages() {
    const B = +bEl.value, T = +tEl.value, C = +cEl.value, H = +hEl.value;
    const hd = Math.floor(C / H);
    return [
      { name: 'input x',              shape: [B, T, C],              note: 'token embeddings + position' },
      { name: 'LayerNorm 1 (x)',      shape: [B, T, C],              note: 'pre-norm, stats per token' },
      { name: 'c_attn(x)',            shape: [B, T, 3*C],            note: 'projects to Q, K, V concatenated' },
      { name: 'Q, K, V split',        shape: [B, T, C],              note: '×3 tensors, each (B,T,C)' },
      { name: 'reshape → heads',      shape: [B, H, T, hd],          note: `split C=${C} into ${H} heads of ${hd}` },
      { name: 'Q @ Kᵀ / √d',           shape: [B, H, T, T],           note: 'attention scores' },
      { name: 'causal mask + softmax', shape: [B, H, T, T],          note: 'per-row softmax' },
      { name: '@ V',                   shape: [B, H, T, hd],          note: 'weighted sum of values' },
      { name: 'reshape ← heads',       shape: [B, T, C],              note: 'concat heads back' },
      { name: 'c_proj(·)',             shape: [B, T, C],              note: 'final attn projection' },
      { name: 'x + attn_out',           shape: [B, T, C],              note: 'residual connection #1' },
      { name: 'LayerNorm 2',            shape: [B, T, C],              note: 'pre-norm for MLP' },
      { name: 'c_fc (→ 4C)',            shape: [B, T, 4*C],            note: 'expand 4× workspace' },
      { name: 'GELU',                   shape: [B, T, 4*C],            note: 'nonlinearity' },
      { name: 'c_proj (4C→C)',          shape: [B, T, C],              note: 'compress back' },
      { name: 'x + mlp_out',            shape: [B, T, C],              note: 'residual connection #2' },
      { name: 'output',                 shape: [B, T, C],              note: 'same shape as input' },
    ];
  }

  function render() {
    const B = +bEl.value, T = +tEl.value, C = +cEl.value, H = +hEl.value;
    container.querySelector('#tb-b-out').textContent = B;
    container.querySelector('#tb-t-out').textContent = T;
    container.querySelector('#tb-c-out').textContent = C;
    container.querySelector('#tb-h-out').textContent = H;
    if (C % H !== 0) { out.innerHTML = '<span style="color:#b91c1c">Warning: C must divide evenly into H heads.</span>'; return; }

    const sg = stages();
    const rowH = 28, yStart = 20;
    let html = '';
    sg.forEach((s, i) => {
      const y = yStart + i * rowH;
      const active = i === highlight;
      const bg = active ? '#fde68a' : (s.name.includes('@') || s.name.includes('softmax') || s.name.includes('norm') ? '#eff6ff' : '#fff');
      html += `<rect x="10" y="${y-12}" width="380" height="${rowH-4}" fill="${bg}" stroke="#d1d5db" rx="3"/>`;
      html += `<text x="20" y="${y+4}" font-size="12" font-family="ui-monospace,monospace" fill="#111">${s.name}</text>`;
      html += `<text x="400" y="${y+4}" font-size="11" font-family="ui-monospace,monospace" fill="#2563eb">(${s.shape.join(', ')})</text>`;
      html += `<text x="540" y="${y+4}" font-size="10" fill="#6b7280">${s.note}</text>`;
      // compute element count
      const numel = s.shape.reduce((a,b) => a*b, 1);
      html += `<text x="790" y="${y+4}" font-size="10" fill="#9ca3af" text-anchor="end">${numel.toLocaleString()} elems</text>`;
    });
    svg.innerHTML = html;

    const totalParams = 4*C*C + 8*C*C; // attn + mlp
    out.innerHTML = `Total params in this block ≈ <b>${totalParams.toLocaleString()}</b> (attention 4C² + MLP 8C²). ` +
      `For GPT-2 small (C=768): ~7M per block × 12 blocks = ~85M.`;
  }

  async function play() {
    const sg = stages();
    for (let i = 0; i < sg.length; i++) {
      highlight = i;
      render();
      await new Promise(r => setTimeout(r, 400));
    }
    highlight = -1;
    render();
  }

  [bEl,tEl,cEl,hEl].forEach(el => el.addEventListener('input', render));
  container.querySelector('#tb-play').addEventListener('click', play);
  render();
});
