// viz/multi_head_attention.js
// Shows the same sequence attended to by 8 heads in parallel.
// Each head has its own pattern; the mini-grid reveals how heads specialize.

registerViz('multi_head_attention', function (container) {
  container.innerHTML = `
    <p class="viz-title">Multi-head attention</p>
    <p class="viz-sub">One input sequence, 8 heads in parallel. Each head develops its own specialization. Hover a head for the pattern it learned.</p>
    <div class="viz-controls">
      <label>Sentence:
        <select id="mha-s">
          <option value="0" selected>The cat sat on the mat quietly</option>
          <option value="1">She gave him the book yesterday</option>
          <option value="2">They went to the park and played</option>
        </select>
      </label>
      <label>Dim per head: <input id="mha-d" type="range" min="8" max="128" step="8" value="64"><span id="mha-d-out">64</span></label>
    </div>
    <div id="mha-grid" style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-top:12px"></div>
    <p class="viz-readout" id="mha-out"></p>
  `;

  const sentences = [
    ['The','cat','sat','on','the','mat','quietly'],
    ['She','gave','him','the','book','yesterday'],
    ['They','went','to','the','park','and','played'],
  ];

  // 8 specialist head patterns. Each function returns attention weights for a query position.
  const heads = [
    { name: 'previous',        color: '#60a5fa', fn: (q, T) => Array.from({length:T}, (_,k) => k === q - 1 ? 0.85 : k === q ? 0.1 : 0.05/T) },
    { name: 'start-of-seq',    color: '#a78bfa', fn: (q, T) => Array.from({length:T}, (_,k) => k === 0 ? 0.7 : k === q ? 0.2 : 0.1/T) },
    { name: 'self',            color: '#34d399', fn: (q, T) => Array.from({length:T}, (_,k) => k === q ? 0.9 : 0.1/T) },
    { name: 'previous-2',      color: '#f59e0b', fn: (q, T) => Array.from({length:T}, (_,k) => k === q - 2 ? 0.7 : k === q ? 0.2 : 0.1/T) },
    { name: 'broadcast',       color: '#ec4899', fn: (q, T) => Array.from({length:T}, (_,k) => 1/T) },
    { name: 'long-range',      color: '#f43f5e', fn: (q, T) => Array.from({length:T}, (_,k) => k < q - 2 ? 0.6/Math.max(1, q - 2) : k === q ? 0.4 : 0) },
    { name: 'the-attends',     color: '#6366f1', fn: (q, T, toks) => Array.from({length:T}, (_,k) => toks[k].toLowerCase() === 'the' ? 0.4 : k === q ? 0.3 : 0.3/T) },
    { name: 'nouns',           color: '#14b8a6', fn: (q, T, toks) => Array.from({length:T}, (_,k) => /^(cat|mat|book|park|she|he|him|they)$/i.test(toks[k]) ? 0.3 : k === q ? 0.2 : 0.2/T) },
  ];

  function normalize(row, q) {
    // apply causal mask (only attend to 0..q) + renormalize
    const causal = row.slice(0, q + 1);
    const rest = new Array(row.length - q - 1).fill(0);
    const sum = causal.reduce((a, b) => a + b, 0) || 1;
    return [...causal.map(v => v / sum), ...rest];
  }

  function render() {
    const idx = +container.querySelector('#mha-s').value;
    const tokens = sentences[idx];
    const T = tokens.length;
    const d = +container.querySelector('#mha-d').value;
    container.querySelector('#mha-d-out').textContent = d;

    const grid = container.querySelector('#mha-grid');
    grid.innerHTML = heads.map((head, h) => {
      // render this head's attention as a TxT grid
      const weights = Array.from({length: T}, (_, q) => normalize(head.fn(q, T, tokens), q));
      const cell = 22;
      const size = T * cell;
      let cells = '';
      for (let i = 0; i < T; i++) {
        for (let j = 0; j < T; j++) {
          const w = weights[i][j];
          const op = Math.min(1, w * 3);
          const bg = j > i ? '#f3f4f6' : `rgba(${parseInt(head.color.slice(1,3),16)}, ${parseInt(head.color.slice(3,5),16)}, ${parseInt(head.color.slice(5,7),16)}, ${op.toFixed(2)})`;
          cells += `<rect x="${j*cell}" y="${i*cell}" width="${cell-1}" height="${cell-1}" fill="${bg}"/>`;
        }
      }
      return `<div style="text-align:center;padding:6px;border:1px solid #e5e7eb;border-radius:6px" title="${head.name}">
        <div style="font-size:11px;font-weight:bold;color:${head.color};margin-bottom:4px">Head ${h+1}: ${head.name}</div>
        <svg viewBox="0 0 ${size} ${size}" width="${size}" height="${size}">${cells}</svg>
      </div>`;
    }).join('');

    const totalParams = heads.length * 3 * d * d;
    container.querySelector('#mha-out').textContent =
      `Each head has its own Q, K, V projections (${d}×${d} each = ${d*d*3} params per head). ` +
      `8 heads × 3 × ${d}² = ${totalParams.toLocaleString()} parameters just for attention. ` +
      `Multiple narrow heads > one wide head, empirically.`;
  }

  container.querySelector('#mha-s').addEventListener('change', render);
  container.querySelector('#mha-d').addEventListener('input', render);
  render();
});
